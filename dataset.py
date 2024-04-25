###
### modified from Nilotpal's code: https://github.com/nilotpal09/HGPflow
###

import uproot
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, Sampler
import itertools

import dgl
import dgl.function as fn
import torch.nn.functional as F

import gc, os

from utils.transformation import VarTransformation
from utils import pdgid_class_dict



class PflowDataset(Dataset):

    def __init__(self, filename, config=None, reduce_ds=-1):
        super().__init__()
	    
        self.filename = filename
        self.config = config
        self.reduce_ds = reduce_ds

        self.init_var_list()
        self.pdgid_to_class = pdgid_class_dict.pdgid_class_dict

        f = uproot.open(filename)
        # self.tree = f['EventTree']
        self.tree = f['Out_Tree']

        self.n_events = self.tree.num_entries
        if (reduce_ds > 0) and (reduce_ds < 1):
            self.n_events = int(self.n_events * reduce_ds)
        elif reduce_ds >= 1:
            self.n_events = min(int(reduce_ds), self.n_events)
        
        self.var_transform = config['var_transform']


        # read the data
        self.data_dict = {}
        for var in tqdm(self.track_vars + self.topo_vars + self.cell_vars + self.particle_vars):
            v_name = self.var_rename[var] if var in self.var_rename else var
            self.data_dict[v_name] = self.tree[var].array(library='np', entry_stop=self.n_events)
            if var in self.var_mev_to_gev or v_name in self.var_mev_to_gev:
                self.data_dict[v_name] /= 1e3

        # convert coordinates of track extrapolated positions from x,y,z to eta,phi
        self.track_extrap_xyz_to_etaphi()


        # adding sin and cos of phis
        for var in config['vars_sin_cos']:
            new_var = var.replace('phi', 'cosphi')
            self.data_dict[new_var] = np.array([np.cos(x.astype(np.float32)) for x in self.data_dict[var]], dtype=object)
            self.data_dict["track_cosphi"] = np.reshape(self.data_dict["track_phi"], (self.data_dict["track_phi"].shape[0], ))

            new_var = var.replace('phi', 'sinphi')
            self.data_dict[new_var] = np.array([np.sin(x.astype(np.float32)) for x in self.data_dict[var]], dtype=object)
            self.data_dict["track_sinphi"] = np.reshape(self.data_dict["track_phi"], (self.data_dict["track_phi"].shape[0], ))


        # create the transform dict
        self.transform_dicts = {}
        for k, v in self.var_transform.items():
            self.transform_dicts[k] = VarTransformation(v)

		# needed for batch sampling
        self.n_nodes = np.array([len(c) + len(t) for c, t in zip(self.data_dict['topo_e'], self.data_dict['track_phi'])])
        self.n_cells = np.array([len(c) for c in self.data_dict['cell_eta']])
        self.n_particles = np.array([len(p) for p in self.data_dict['particle_pt']])

        print('\ndataset loaded')
        print(f'number of events: {self.n_events}\n')



    def __getitem__(self, idx):

        # read all the variables (tracks)
        track_qoverp_raw  = torch.tensor(self.data_dict['track_qoverp'][idx], dtype=torch.float)
        track_theta_raw = torch.tensor(self.data_dict['track_theta'][idx], dtype=torch.float)
        track_phi     = torch.tensor(self.data_dict['track_phi'][idx], dtype=torch.float)
        track_cosphi  = torch.tensor(self.data_dict['track_cosphi'][idx], dtype=torch.float)
        track_sinphi  = torch.tensor(self.data_dict['track_sinphi'][idx], dtype=torch.float)
        track_d0      = torch.tensor(self.data_dict['track_d0'][idx], dtype=torch.float)
        track_z0      = torch.tensor(self.data_dict['track_z0'][idx], dtype=torch.float)

        # track var reparameterization
        track_pt_raw    = torch.abs(1/track_qoverp_raw) * torch.sin(track_theta_raw) / 1e3
        track_eta_raw   = -torch.log(torch.tan(track_theta_raw/2))

        # read all the variables (cells)
        cell_eta_raw = torch.tensor(self.data_dict['cell_eta'][idx], dtype=torch.float)
        cell_phi     = torch.tensor(self.data_dict['cell_phi'][idx], dtype=torch.float)
        cell_cosphi  = torch.tensor(self.data_dict['cell_cosphi'][idx], dtype=torch.float)
        cell_sinphi  = torch.tensor(self.data_dict['cell_sinphi'][idx], dtype=torch.float)
        cell_e_raw   = torch.tensor(self.data_dict['cell_e'][idx], dtype=torch.float)
        cell_calo_region = torch.tensor(self.data_dict['cell_calo_region'][idx], dtype=torch.long)

        cell_x = torch.tensor(self.data_dict['cell_x'][idx], dtype=torch.float)
        cell_y = torch.tensor(self.data_dict['cell_y'][idx], dtype=torch.float)
        cell_z = torch.tensor(self.data_dict['cell_z'][idx], dtype=torch.float)
        # cell_pos = torch.tensor(self.data_dict)
        

        # read all the variables (topos)
        topo_e_raw   = torch.tensor(self.data_dict['topo_e'][idx], dtype=torch.float)
        topo_eta_raw = torch.tensor(self.data_dict['topo_eta'][idx], dtype=torch.float)
        topo_phi     = torch.tensor(self.data_dict['topo_phi'][idx], dtype=torch.float)
        topo_cosphi  = torch.tensor(self.data_dict['topo_cosphi'][idx], dtype=torch.float)
        topo_sinphi  = torch.tensor(self.data_dict['topo_sinphi'][idx], dtype=torch.float)
        topo_rho     = torch.tensor(self.data_dict['topo_rho'][idx], dtype=torch.float)
        topo_sigma_eta = torch.tensor(self.data_dict['topo_sigma_eta'][idx], dtype=torch.float)
        topo_sigma_phi = torch.tensor(self.data_dict['topo_sigma_phi'][idx], dtype=torch.float)


        # read all the variables (particles)
        particle_pt    = torch.tensor(self.data_dict['particle_pt'][idx], dtype=torch.float)
        particle_eta   = torch.tensor(self.data_dict['particle_eta'][idx], dtype=torch.float)
        particle_phi   = torch.tensor(self.data_dict['particle_phi'][idx], dtype=torch.float)
        particle_class = torch.LongTensor([self.pdgid_to_class[x] for x in self.data_dict['particle_pdgid'][idx]])
        particle_e_raw = torch.tensor(self.data_dict['particle_e'][idx], dtype=torch.float)


        # parent particle links
        cell_particle_idx = torch.tensor(self.data_dict['cell_particle_idx'][idx], dtype=torch.long)
        track_particle_idx = torch.tensor(self.data_dict['track_particle_idx'][idx], dtype=torch.long)


        # transformations
        track_pt  = self.transform_dicts['pt'].forward(track_pt_raw)
        track_eta = self.transform_dicts['eta'].forward(track_eta_raw)
        track_d0  = self.transform_dicts['d0'].forward(track_d0)
        track_z0  = self.transform_dicts['z0'].forward(track_z0)

        cell_e   = self.transform_dicts['e'].forward(cell_e_raw)
        cell_eta = self.transform_dicts['eta'].forward(cell_eta_raw)
        cell_x   = self.transform_dicts['x'].forward(cell_x)
        cell_y   = self.transform_dicts['y'].forward(cell_y)
        cell_z   = self.transform_dicts['z'].forward(cell_z)

        topo_e   = self.transform_dicts['e'].forward(topo_e_raw)
        topo_eta = self.transform_dicts['eta'].forward(topo_eta_raw)
        topo_sigma_eta = self.transform_dicts['sigma_eta'].forward(topo_sigma_eta)
        topo_sigma_phi = self.transform_dicts['sigma_phi'].forward(topo_sigma_phi)

        particle_pt  = self.transform_dicts['pt'].forward(particle_pt)
        particle_eta = self.transform_dicts['eta'].forward(particle_eta)


        n_tracks     = len(track_eta)
        n_cells      = len(cell_eta)
        n_topos      = len(topo_eta)
        n_nodes      = n_tracks + n_cells
        n_particles  = len(particle_pt)



        num_nodes_dict = {
            'track': n_tracks,
            'cell': n_cells,
            'topo': n_topos,

            'pre_node': n_tracks + n_cells,
            'node': n_nodes,

            'particle': n_particles,
            'pflow_particle': n_particles,

            'global_node': 1
        }

        # pre_node = [tracks, cells]
        if self.config['graph_building'] == 'predefined':

            # assert that max(src and dst) < n_cells
            if len(self.data_dict['cell_to_cell_src'][idx]) > 0:
                assert self.data_dict['cell_to_cell_src'][idx].max() <= n_cells 
            if len(self.data_dict['cell_to_cell_dst'][idx]) > 0:
                assert self.data_dict['cell_to_cell_dst'][idx].max() <= n_cells
            if len(self.data_dict['track_to_cell_dst'][idx]) > 0:
                assert self.data_dict['track_to_cell_dst'][idx].max() <= n_cells 

            # edges read from the file
            pre_node_to_pre_node_start = torch.cat([
                torch.tensor(self.data_dict['track_to_cell_src'][idx], dtype=torch.long),
                torch.tensor(self.data_dict['track_to_cell_dst'][idx], dtype=torch.long) + n_tracks, # comment out
                torch.tensor(self.data_dict['cell_to_cell_src'][idx], dtype=torch.long) + n_tracks
            ], dim=0)
            pre_node_to_pre_node_end   = torch.cat([
                torch.tensor(self.data_dict['track_to_cell_dst'][idx], dtype=torch.long) + n_tracks,
                torch.tensor(self.data_dict['track_to_cell_src'][idx], dtype=torch.long), # comment out
                torch.tensor(self.data_dict['cell_to_cell_dst'][idx], dtype=torch.long) + n_tracks
            ], dim=0)

        elif self.config['graph_building'] == 'knn':

            # distance matrix
            # diag: track-track = None; cell-cell = d(x, y, z)
            # off-diag: track-cell = d(eta, phi); cell-track = d(eta, phi)

            # cells to cells (x,y,z)
            cc_dists = np.sqrt(
                (cell_x[:, None] - cell_x[None, :])**2 + \
                (cell_y[:, None] - cell_y[None, :])**2 + \
                (cell_z[:, None] - cell_z[None, :])**2)

            k = min(self.config['knn_k_cell'], n_cells)
            values, indices = torch.topk(cc_dists, k, dim=1, largest=False)

            # The indices represent the end nodes for each start node
            cell_to_cell_start = torch.arange(cc_dists.shape[0]).unsqueeze(-1).repeat(1, k).view(-1)
            cell_to_cell_end = indices.view(-1)


            # tracks to cells (eta, phi)
            tc_dists = self.delta_r(
                track_eta_raw[:, None], cell_eta_raw[None, :], track_phi[:, None], cell_phi[None, :])

            k = min(self.config['knn_k_track_cell'], n_cells)
            values, indices = torch.topk(tc_dists, k, dim=1, largest=False)

            # The indices represent the end nodes for each start node
            track_to_cell_start = torch.arange(tc_dists.shape[0]).unsqueeze(-1).repeat(1, k).view(-1)
            track_to_cell_end = indices.view(-1)


            # combine the edges
            pre_node_to_pre_node_start = torch.cat([
                cell_to_cell_start + n_tracks, # cell-cell
                track_to_cell_start, # track-cell
                track_to_cell_end + n_tracks # track-cell flipped
            ], dim=0)
            pre_node_to_pre_node_end = torch.cat([
                cell_to_cell_end + n_tracks, # cell-cell
                track_to_cell_end + n_tracks, # track-cell
                track_to_cell_start # track-cell flipped
            ], dim=0)

        elif self.config['graph_building'] == 'knn+predefined':
                
            # distance matrix
            # diag: track-track = None; cell-cell = predefined
            # off-diag: track-cell = d(eta, phi); cell-track = d(eta, phi)

            # assert that max(src and dst) < n_cells
            if len(self.data_dict['cell_to_cell_src'][idx]) > 0:
                assert self.data_dict['cell_to_cell_src'][idx].max() <= n_cells 
            if len(self.data_dict['cell_to_cell_dst'][idx]) > 0:
                assert self.data_dict['cell_to_cell_dst'][idx].max() <= n_cells

            # tracks to cells (eta, phi)
            tc_dists = self.delta_r(
                track_eta_raw[:, None], cell_eta_raw[None, :], track_phi[:, None], cell_phi[None, :])

            k = min(self.config['knn_k_track_cell'], n_cells)
            values, indices = torch.topk(tc_dists, k, dim=1, largest=False)

            # The indices represent the end nodes for each start node
            track_to_cell_start = torch.arange(tc_dists.shape[0]).unsqueeze(-1).repeat(1, k).view(-1)
            track_to_cell_end = indices.view(-1)


            # combine the edges
            pre_node_to_pre_node_start = torch.cat([
                torch.tensor(self.data_dict['cell_to_cell_src'][idx], dtype=torch.long) + n_tracks, # cell-cell
                track_to_cell_start, # track-cell
                track_to_cell_end + n_tracks # track-cell flipped
            ], dim=0)
            pre_node_to_pre_node_end = torch.cat([
                torch.tensor(self.data_dict['cell_to_cell_dst'][idx], dtype=torch.long) + n_tracks, # cell-cell
                track_to_cell_end + n_tracks, # track-cell
                track_to_cell_start # track-cell flipped
            ], dim=0)

        else:
            raise ValueError(f'graph_building {self.config["graph_building"]} not supported')
        

        if self.config['detector'] == 'ATLAS':
            pre_node_to_topo_start = torch.tensor(self.data_dict['cell_topo_start'][idx], dtype=torch.long) + n_tracks
            pre_node_to_topo_end   = torch.tensor(self.data_dict['cell_topo_end'][idx], dtype=torch.long)
            pre_node_to_topo_e_raw = torch.tensor(self.data_dict['cell_topo_e'][idx], dtype=torch.float)


        data_dict = {
			('node','node_to_node','node') : (pre_node_to_pre_node_start, pre_node_to_pre_node_end),
        }

        g = dgl.heterograph(data_dict, num_nodes_dict)

        # decorate the graph with features
        g.nodes['cell'].data['x']           = cell_x
        g.nodes['cell'].data['y']           = cell_y
        g.nodes['cell'].data['z']           = cell_z
        g.nodes['cell'].data['eta']         = cell_eta
        g.nodes['cell'].data['phi']         = cell_phi
        g.nodes['cell'].data['cosphi']      = cell_cosphi
        g.nodes['cell'].data['sinphi']      = cell_sinphi
        g.nodes['cell'].data['e']           = cell_e_raw
        g.nodes['cell'].data['calo_region'] = cell_calo_region
        g.nodes['cell'].data['topo_idx']    = torch.tensor(self.data_dict['cell_topo_idx'][idx], dtype=torch.long)

        g.nodes['cell'].data['x'] = cell_x
        g.nodes['cell'].data['y'] = cell_y
        g.nodes['cell'].data['z'] = cell_z
        

        g.nodes['track'].data['pt']     = track_pt
        g.nodes['track'].data['eta']    = track_eta
        g.nodes['track'].data['phi']    = track_phi
        g.nodes['track'].data['cosphi'] = track_cosphi
        g.nodes['track'].data['sinphi'] = track_sinphi
        g.nodes['track'].data['d0']     = track_d0
        g.nodes['track'].data['z0']     = track_z0

        for i in range(6):
            var = f'cosphi_layer_{i}'
            g.nodes['track'].data[var] = torch.tensor(self.data_dict[f'track_{var}'][idx].astype(np.float32), dtype=torch.float)

            var = f'sinphi_layer_{i}'
            g.nodes['track'].data[var] = torch.tensor(self.data_dict[f'track_{var}'][idx].astype(np.float32), dtype=torch.float)

            var = f'eta_layer_{i}'
            tmp_eta_raw = torch.tensor(self.data_dict[f'track_{var}'][idx], dtype=torch.float)
            g.nodes['track'].data[var] = self.transform_dicts['eta'].forward(tmp_eta_raw)

        g.nodes['topo'].data['phi']    = topo_phi
        g.nodes['topo'].data['cosphi'] = topo_cosphi
        g.nodes['topo'].data['sinphi'] = topo_sinphi
        g.nodes['topo'].data['eta']    = topo_eta
        g.nodes['topo'].data['e']      = topo_e
        g.nodes['topo'].data['rho']    = topo_rho

        g.nodes['node'].data['eta']      = torch.cat([track_eta, cell_eta], dim=0)
        g.nodes['node'].data['phi']      = torch.cat([track_phi, cell_phi], dim=0)
        g.nodes['node'].data['cosphi']   = torch.cat([track_cosphi, cell_cosphi], dim=0)
        g.nodes['node'].data['sinphi']   = torch.cat([track_sinphi, cell_sinphi], dim=0)
        g.nodes['node'].data['calo_region'] = torch.cat([torch.zeros_like(track_pt, dtype=torch.long), cell_calo_region], dim=0)
        g.nodes['node'].data['e']        = torch.cat([torch.zeros_like(track_pt), cell_e_raw], dim=0)
        g.nodes['node'].data['pt']       = torch.cat([track_pt, torch.zeros_like(cell_eta)], dim=0)
        g.nodes['node'].data['is_track'] = torch.cat([torch.ones_like(track_pt), torch.zeros_like(cell_eta)], dim=0)
        g.nodes['node'].data['is_cell']  = torch.cat([torch.zeros_like(track_pt), torch.ones_like(cell_eta)], dim=0)
        g.nodes['node'].data['particle_idx'] = torch.cat([track_particle_idx, cell_particle_idx], dim=0)

        # easy to access later
        g.nodes['node'].data['eta_raw'] = torch.cat([track_eta_raw, cell_eta_raw], dim=0)
        g.nodes['node'].data['phi_raw'] = torch.cat([track_phi, cell_phi], dim=0)
        g.nodes['node'].data['e_raw']   = torch.cat([torch.zeros_like(track_pt), cell_e_raw], dim=0)
        g.nodes['node'].data['pt_raw']  = torch.cat([track_pt_raw, torch.zeros_like(cell_e_raw)], dim=0)

        g.nodes['particle'].data['pt']     = particle_pt
        g.nodes['particle'].data['eta']    = particle_eta
        g.nodes['particle'].data['phi']    = particle_phi
        g.nodes['particle'].data['e_raw']  = particle_e_raw

        particle_is_neut_mask = (particle_class >= 3)
        track_particle_idx = torch.tensor(self.data_dict['track_particle_idx'][idx], dtype=torch.int64)
        good_class_particles = torch.scatter(
            particle_is_neut_mask, 0, track_particle_idx, True)
        ch_particles_with_no_track_mask = ~good_class_particles

        # add +3 to the class of the charged particles with no track, given they are not muons
        particle_class[ch_particles_with_no_track_mask * (particle_class!=2)] += 3

        # trackess muons will be photons
        particle_class[(particle_class==2) * ch_particles_with_no_track_mask] = 4

        g.nodes['particle'].data['class'] = particle_class


        # any additional COCOA-only related variables
        if self.config['detector'] == 'COCOA':
            for i in range(6):
                var = f'cosphi_layer_{i}'
                g.nodes['track'].data[var] = torch.tensor(self.data_dict[f'track_{var}'][idx].astype(np.float32), dtype=torch.float)

                var = f'sinphi_layer_{i}'
                g.nodes['track'].data[var] = torch.tensor(self.data_dict[f'track_{var}'][idx].astype(np.float32), dtype=torch.float)

                var = f'eta_layer_{i}'
                tmp_eta_raw = torch.tensor(self.data_dict[f'track_{var}'][idx], dtype=torch.float)
                g.nodes['track'].data[var] = self.transform_dicts['eta'].forward(tmp_eta_raw)

            # g.nodes['topo'].data['raw_e_ecal'] = torch.tensor(self.data_dict['topo_ecal_e'][idx], dtype=torch.float)
            # g.nodes['topo'].data['raw_e_hcal'] = torch.tensor(self.data_dict['topo_hcal_e'][idx], dtype=torch.float)
            # g.nodes['topo'].data['em_frac'] = g.nodes['topo'].data['raw_e_ecal'] / \
            #     (g.nodes['topo'].data['raw_e_ecal'] + g.nodes['topo'].data['raw_e_hcal'])

            # g.nodes['node'].data['em_frac'] = torch.cat([torch.zeros_like(track_pt), g.nodes['topo'].data['em_frac']], dim=0)

        # any additional CLIC-only related variables
        elif self.config['detector'] == 'CLIC':
            g.nodes['track'].data['chi2'] = torch.tensor(self.data_dict['track_chi2'][idx], dtype=torch.float)
            g.nodes['track'].data['ndf'] = torch.tensor(self.data_dict['track_ndf'][idx], dtype=torch.float)
            g.nodes['track'].data['radiusofinnermosthit'] = torch.tensor(self.data_dict['track_radiusofinnermosthit'][idx], dtype=torch.float)
            g.nodes['track'].data['tanlambda'] = torch.tensor(self.data_dict['track_tanlambda'][idx], dtype=torch.float)
            g.nodes['track'].data['omega'] = torch.tensor(self.data_dict['track_omega'][idx], dtype=torch.float)

            g.nodes['cell'].data['rho'] = self.transform_dicts['rho'].forward(
                torch.tensor(self.data_dict['cell_rho'][idx], dtype=torch.float))
            g.nodes['topo'].data['sigma_rho'] = self.transform_dicts['sigma_rho'].forward(
                torch.tensor(self.data_dict['topo_sigma_rho'][idx], dtype=torch.float))
            g.nodes['topo'].data['raw_e_ecal'] = torch.tensor(
                self.data_dict['topo_energy_ecal'][idx], dtype=torch.float)
            g.nodes['topo'].data['raw_e_hcal'] = torch.tensor(
                self.data_dict['topo_energy_hcal'][idx], dtype=torch.float)
            g.nodes['topo'].data['raw_e_other'] = torch.tensor(
                self.data_dict['topo_energy_other'][idx], dtype=torch.float)
            g.nodes['topo'].data['em_frac'] = g.nodes['topo'].data['raw_e_ecal'] / \
                (g.nodes['topo'].data['raw_e_ecal'] + g.nodes['topo'].data['raw_e_hcal'] + g.nodes['topo'].data['raw_e_other'])

            g.nodes['node'].data['sigma_rho'] = torch.cat([torch.zeros_like(track_pt), g.nodes['topo'].data['sigma_rho']], dim=0)
            g.nodes['node'].data['em_frac'] = torch.cat([torch.zeros_like(track_pt), g.nodes['topo'].data['em_frac']], dim=0)

        elif self.config['detector'] == 'ATLAS':
            for t_i in range(1, 4):
                g.nodes['track'].data[f'eta_emb{t_i}'] = torch.tensor(self.data_dict[f'track_eta_emb{t_i}'][idx], dtype=torch.float)
                g.nodes['track'].data[f'cosphi_emb{t_i}'] = torch.tensor(self.data_dict[f'track_cosphi_emb{t_i}'][idx], dtype=torch.float)
                g.nodes['track'].data[f'sinphi_emb{t_i}'] = torch.tensor(self.data_dict[f'track_sinphi_emb{t_i}'][idx], dtype=torch.float)

            for t_i in range(1, 4):
                g.nodes['track'].data[f'eta_eme{t_i}'] = torch.tensor(self.data_dict[f'track_eta_eme{t_i}'][idx], dtype=torch.float)
                g.nodes['track'].data[f'cosphi_eme{t_i}'] = torch.tensor(self.data_dict[f'track_cosphi_eme{t_i}'][idx], dtype=torch.float)
                g.nodes['track'].data[f'sinphi_eme{t_i}'] = torch.tensor(self.data_dict[f'track_sinphi_eme{t_i}'][idx], dtype=torch.float)

            for t_i in range(0, 4):
                g.nodes['track'].data[f'eta_hec{t_i}'] = torch.tensor(self.data_dict[f'track_eta_hec{t_i}'][idx], dtype=torch.float)
                g.nodes['track'].data[f'cosphi_hec{t_i}'] = torch.tensor(self.data_dict[f'track_cosphi_hec{t_i}'][idx], dtype=torch.float)
                g.nodes['track'].data[f'sinphi_hec{t_i}'] = torch.tensor(self.data_dict[f'track_sinphi_hec{t_i}'][idx], dtype=torch.float)

            for t_i in range(0, 3):
                g.nodes['track'].data[f'eta_tilebar{t_i}'] = torch.tensor(self.data_dict[f'track_eta_tilebar{t_i}'][idx], dtype=torch.float)
                g.nodes['track'].data[f'cosphi_tilebar{t_i}'] = torch.tensor(self.data_dict[f'track_cosphi_tilebar{t_i}'][idx], dtype=torch.float)
                g.nodes['track'].data[f'sinphi_tilebar{t_i}'] = torch.tensor(self.data_dict[f'track_sinphi_tilebar{t_i}'][idx], dtype=torch.float)

            for t_i in range(1, 4):
                g.nodes['track'].data[f'eta_tilegap{t_i}'] = torch.tensor(self.data_dict[f'track_eta_tilegap{t_i}'][idx], dtype=torch.float)
                g.nodes['track'].data[f'cosphi_tilegap{t_i}'] = torch.tensor(self.data_dict[f'track_cosphi_tilegap{t_i}'][idx], dtype=torch.float)
                g.nodes['track'].data[f'sinphi_tilegap{t_i}'] = torch.tensor(self.data_dict[f'track_sinphi_tilegap{t_i}'][idx], dtype=torch.float)

            for t_i in range(0, 3):
                g.nodes['track'].data[f'eta_tileext{t_i}'] = torch.tensor(self.data_dict[f'track_eta_tileext{t_i}'][idx], dtype=torch.float)
                g.nodes['track'].data[f'cosphi_tileext{t_i}'] = torch.tensor(self.data_dict[f'track_cosphi_tileext{t_i}'][idx], dtype=torch.float)
                g.nodes['track'].data[f'sinphi_tileext{t_i}'] = torch.tensor(self.data_dict[f'track_sinphi_tileext{t_i}'][idx], dtype=torch.float)

            g.nodes['cell'].data['rho'] = self.transform_dicts['rho'].forward(
                torch.tensor(self.data_dict['cell_rho'][idx], dtype=torch.float))

            g.nodes['topo'].data['em_frac'] = torch.tensor(self.data_dict['topo_em_frac'][idx], dtype=torch.float)
            g.nodes['topo'].data['sigma_rho'] = self.transform_dicts['sigma_rho'].forward(
                torch.tensor(self.data_dict['topo_sigma_rho'][idx], dtype=torch.float))

            g.nodes['node'].data['em_frac'] = torch.cat([torch.zeros_like(track_pt), g.nodes['topo'].data['em_frac']], dim=0)
            g.nodes['node'].data['sigma_rho'] = torch.cat([torch.zeros_like(track_pt), g.nodes['topo'].data['sigma_rho']], dim=0)

            # edges values
            g.edges['pre_node_to_topo'].data['raw_e'] = pre_node_to_topo_e_raw


        return g, idx




    def __len__(self):
        return self.n_events

    def init_var_list(self):
        self.track_vars = self.config['vars_to_read']['track_vars']
        self.topo_vars  = self.config['vars_to_read']['topo_vars']
        self.cell_vars  = self.config['vars_to_read']['cell_vars']
        self.particle_vars = self.config['vars_to_read']['particle_vars']

        self.var_rename = self.config['var_rename']
        self.var_mev_to_gev = self.config['var_mev_to_gev']

    def delta_r(self, eta1, eta2, phi1, phi2, rho1=None, rho2=None):
        dphi = (phi1 - phi2 + np.pi) % (2 * np.pi) - np.pi
        deta = eta1 - eta2
        if rho1 is not None and rho2 is not None:
            drho = rho1 - rho2
            return np.sqrt(deta**2 + dphi**2 + drho**2)
        else:
            return np.sqrt(deta**2 + dphi**2)

    def track_extrap_xyz_to_etaphi(self):

        # doing the empty stuff, otherwise setting array.dtype=obj will force the element to be object (not float) if the array is kinda homogeneous
        for layer in range(6):
            rho = np.empty_like(self.data_dict[f'track_z_layer_{layer}'])
            rho[:] = [np.sqrt(x**2 + y**2) for x, y in zip(self.data_dict[f'track_x_layer_{layer}'], self.data_dict[f'track_y_layer_{layer}'])]

            r = np.empty_like(self.data_dict[f'track_z_layer_{layer}'])
            r[:] = [np.sqrt(rho_**2 + z_**2) for rho_, z_ in zip(rho, self.data_dict[f'track_z_layer_{layer}'])]

            theta = np.empty_like(self.data_dict[f'track_z_layer_{layer}'])
            theta[:] = [np.arcsin(rho_/r_) for rho_, r_ in zip(rho, r)]

            self.data_dict[f'track_eta_layer_{layer}'] = np.empty_like(self.data_dict[f'track_z_layer_{layer}'])
            self.data_dict[f'track_eta_layer_{layer}'][:] = [-np.log(np.tan(theta_/2)) for theta_ in theta]
            
            self.data_dict[f'track_phi_layer_{layer}'] = np.empty_like(self.data_dict[f'track_z_layer_{layer}'])
            self.data_dict[f'track_phi_layer_{layer}'][:] = [np.arctan2(y_,x_) for x_, y_ in zip(self.data_dict[f'track_x_layer_{layer}'], self.data_dict[f'track_y_layer_{layer}'])]
    

def collate_graphs(samples):

    bs = len(samples)

    batch_num_nodes_cell = [x[0].number_of_nodes('cell') for x in samples]
    max_num_nodes_cell   = max(batch_num_nodes_cell)

    batch_num_nodes_track = [x[0].number_of_nodes('track') for x in samples]
    max_num_nodes_track   = max(batch_num_nodes_track)

    batch_num_nodes_topo = [x[0].number_of_nodes('topo') for x in samples]
    max_num_nodes_topo   = max(batch_num_nodes_topo)

    batch_num_nodes_node = [x[0].number_of_nodes('node') for x in samples]
    first_element = batch_num_nodes_node[0]
    assert all(element == first_element for element in batch_num_nodes_node), \
        f"Assertion error in collate_graphs: " + \
        "all events in a batch must have the same number of nodes (tracks + topo)" + \
        f"idxs: {[x[3] for x in samples]}"
    num_nodes_node = first_element

    # it's set to max_num_particles)
    num_nodes_particle = samples[0][0].number_of_nodes('particle')


    # reference for types etc
    g_0 = samples[0][0]
    feat_dict  = {}


    # init the node features to zero
    cell_feat_names = samples[0][0].nodes['cell'].data.keys()
    for name in cell_feat_names:
        feat_dict[f'cell_{name}'] = torch.zeros(bs, max_num_nodes_cell, dtype=g_0.nodes['cell'].data[name].dtype)
    feat_dict['cell_q_mask'] = torch.zeros(bs, max_num_nodes_cell, dtype=torch.bool)

    track_feat_names = samples[0][0].nodes['track'].data.keys()
    for name in track_feat_names:
        feat_dict[f'track_{name}'] = torch.zeros(bs, max_num_nodes_track, dtype=g_0.nodes['track'].data[name].dtype)
    feat_dict['track_q_mask'] = torch.zeros(bs, max_num_nodes_track, dtype=torch.bool)

    topo_feat_names = samples[0][0].nodes['topo'].data.keys()
    for name in topo_feat_names:
        feat_dict[f'topo_{name}'] = torch.zeros(bs, max_num_nodes_topo, dtype=g_0.nodes['topo'].data[name].dtype)
    feat_dict['topo_q_mask'] = torch.zeros(bs, max_num_nodes_topo, dtype=torch.bool)

    node_feat_names = samples[0][0].nodes['node'].data.keys()
    for name in node_feat_names:
        feat_dict[f'node_{name}'] = torch.zeros(bs, num_nodes_node, dtype=g_0.nodes['node'].data[name].dtype)

    particle_feat_names = samples[0][0].nodes['particle'].data.keys()
    for name in particle_feat_names:
        feat_dict[f'particle_{name}'] = torch.zeros(bs, num_nodes_particle, dtype=g_0.nodes['particle'].data[name].dtype)
    
    # THIS IS POINTLESS as the NUM_PARTICLES is set to MAX_NUM_PARTICLES in the graph
    # feat_dict['particle_q_mask'] = torch.zeros(bs, num_nodes_particle, dtype=torch.bool)



    # init the edge features to zero
    feat_dict['pre_node_to_pre_node_edge_mask']  = torch.zeros(
        bs, max_num_nodes_track + max_num_nodes_cell, max_num_nodes_track + max_num_nodes_cell, dtype=torch.bool)
    
    feat_dict['pre_node_to_topo_edge_mask'] = torch.zeros(
        bs, max_num_nodes_track + max_num_nodes_cell, max_num_nodes_topo, dtype=torch.bool)



    # fill in the values
    for i, (g, _, _, _, _) in enumerate(samples):

        # nodes first
        for n in cell_feat_names:
            feat_dict[f'cell_{n}'][i, :batch_num_nodes_cell[i]] = g.nodes['cell'].data[n]
        feat_dict['cell_q_mask'][i, :batch_num_nodes_cell[i]] = True

        for n in track_feat_names:
            feat_dict[f'track_{n}'][i, :batch_num_nodes_track[i]] = g.nodes['track'].data[n]
        feat_dict['track_q_mask'][i, :batch_num_nodes_track[i]] = True

        for n in topo_feat_names:
            feat_dict[f'topo_{n}'][i, :batch_num_nodes_topo[i]] = g.nodes['topo'].data[n]
        feat_dict['topo_q_mask'][i, :batch_num_nodes_topo[i]] = True

        for n in node_feat_names:
            feat_dict[f'node_{n}'][i, :num_nodes_node] = g.nodes['node'].data[n]

        for n in particle_feat_names:
            feat_dict[f'particle_{n}'][i, :num_nodes_particle] = g.nodes['particle'].data[n]
        # feat_dict['particle_q_mask'][i, :num_nodes_particle] = True


        # now edges
        src, dst = g.edges(etype='pre_node_to_pre_node')
        src_mask = src >= batch_num_nodes_track[i]
        src[src_mask] = src[src_mask] + max_num_nodes_track - batch_num_nodes_track[i]
        dst_mask = dst >= batch_num_nodes_track[i]
        dst[dst_mask] = dst[dst_mask] + max_num_nodes_track - batch_num_nodes_track[i]
        feat_dict['pre_node_to_pre_node_edge_mask'][i, src, dst] = True

        src, dst = g.edges(etype='pre_node_to_topo')
        src_mask = src >= batch_num_nodes_track[i]
        src[src_mask] = src[src_mask] + max_num_nodes_track - batch_num_nodes_track[i]
        feat_dict['pre_node_to_topo_edge_mask'][i, src, dst] = True

    for k, v in feat_dict.items():
        if 'mask' not in k:
            feat_dict[k] = v.unsqueeze(-1)

    feat_dict['idx'] = np.array([x[3] for x in samples])

    return feat_dict

        

class PflowSampler(Sampler):
    def __init__(self, n_nodes_array, batch_size, n_cells_array=None, n_cells_threshold=1700, remove_idxs=[]):
        """
        Initialization
        :param n_nodes_array: array of the number of nodes (tracks + topos)
        :param batch_size: batch size
        """
        super().__init__(n_nodes_array.size)

        self.dataset_size = n_nodes_array.size
        self.batch_size = batch_size

        self.drop_last = False

        self.index_to_batch = {}
        self.node_size_idx = {}
        running_idx = -1

        for n_nodes_i in set(n_nodes_array):

            self.node_size_idx[n_nodes_i] = np.where(n_nodes_array == n_nodes_i)[0]
            self.node_size_idx[n_nodes_i] = np.setdiff1d(self.node_size_idx[n_nodes_i], remove_idxs)
            
            indices = np.arange (0, len(self.node_size_idx[n_nodes_i]), self.batch_size)
            self.node_size_idx[n_nodes_i] = [self.node_size_idx[n_nodes_i][i: i + self.batch_size] for i in indices]

            for batch in self.node_size_idx[n_nodes_i]:
                running_idx += 1

                do_split = False
                if (n_cells_array is not None) and (n_cells_array[batch].max() > n_cells_threshold):
                    opt_bs = self.batch_size * (n_cells_threshold / n_cells_array[batch].max())**2
                    n_split_batches = int(np.ceil(len(batch) / opt_bs))
                    if n_split_batches > 1:
                        do_split = True

                if do_split:
                    batches = np.array_split(batch, n_split_batches)
                    for i, b in enumerate(batches):
                        self.index_to_batch[running_idx] = b
                        if i != n_split_batches - 1:
                            running_idx += 1

                    # batch1, batch2 = np.array_split(batch, 2)
                    # self.index_to_batch[running_idx] = batch1
                    # running_idx += 1
                    # self.index_to_batch[running_idx] = batch2
                else:
                    self.index_to_batch[running_idx] = batch

        self.n_batches = running_idx + 1

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        batch_order = np.random.permutation(np.arange(self.n_batches))
        for i in batch_order:
            yield self.index_to_batch[i]