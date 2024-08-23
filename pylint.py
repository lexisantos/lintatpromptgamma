# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 15:11:17 2024

@author: Alex
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime
import copy
from itertools import combinations
from scipy.stats import chi2


class data_sead:
    def __init__(self, path_SEAD, fecha_ensayo):
        data_sead = pd.read_csv(path_SEAD, delimiter =',')
        data_sead['Hora'] = list(datetime.strptime(fecha_ensayo + '_'+ hora, '%d%b%Y_%H:%M:%S.%f') for hora in data_sead['Hora'])
        t_inicio_sead = data_sead['Hora'][0]
        data_sead_time = np.array([(data_sead_time - t_inicio_sead).total_seconds() for data_sead_time in data_sead['Hora']])
        self.data = data_sead
        self.time = data_sead_time/60
        self.t_inicio = t_inicio_sead
        self.fecha = fecha_ensayo
    
    def copy(self):
        return copy.deepcopy(self)
    
    def CIC_Marcha(self, det):
        current = self.data['LOG {}'.format(str(det))]
        return current
        
    def select_steps(self, det):
        t = self.time 
        i = self.CIC_Marcha(self, det)
        plt.figure(99)
        plt.plot(t, i, '.', label = str(det))
        # plt.plot(data_sead_time, data_sead['LOG M3'], '.', label = 'M3')
        plt.suptitle('Exported data from SEAD. Start time = {}'.format(self.t_inicio))
        plt.xlabel('t [min]')
        plt.ylabel('Current [A]')
        plt.legend()
        plt.grid(ls='--')
        plt.yscale('log')
        plt.tight_layout()
        x = plt.ginput(n=-1, timeout=0)
        step_index = np.row_stack(np.split(np.array([xx[0] for xx in x]), len(x)//2))
        np.savetxt(f'step_t_min_{self.fecha}_consubida.txt', step_index, delimiter='\t')
        plt.close(99)
        return step_index
    
    def per_step(self, det, path_step = None, power_list = list(range(50))):
        try:
            step_times = np.loadtxt(path_step)
        except:
            step_times = self.select_steps(self, det)
        data_step = {}
        time_step = {}

        for jj, interval in enumerate(step_times):
            cond_sead = np.logical_and(self.time>interval[0], self.time<interval[1])
            data_step[power_list[jj]] = self.CIC_Marcha(det)[cond_sead]
            time_step[power_list[jj]] = self.time[cond_sead]
        return time_step, data_step
    
    
def process_lint(path, t_inicio):
    data_lint = pd.read_table(path, delimiter =';|\s+', header=None, engine = 'python')
    data_time = data_lint.get([x for x in range(6)])
    time = np.array([(datetime(*data_time.iloc[i].to_list()) - t_inicio).total_seconds() for i in range(len(data_time))])
    time = time/60
    counts = data_lint[6]
    cod = data_lint[7]
    return time, counts, cod

class data_LINT:
    def __init__(self, path_LINT_list, path_step, t_inicio, save_data=False, power_list = False):
        step_times = np.loadtxt(path_step)
        if power_list:
            potencias = power_list
        else:
            potencias = list(range(len(step_times)))
        step_times = np.loadtxt(path_step)
        counts_cond = {ii:[] for ii in potencias}
        time_cond = {ii:[] for ii in potencias}
        cod_Ts = {ii:[] for ii in potencias}
        for path in path_LINT_list:
            time, counts, cod = process_lint(path, t_inicio)
            for jj, interval in enumerate(step_times):
                cond = np.logical_and(time>interval[0], time<interval[1])
                if any(cond) == True:    
                    counts_cond[potencias[jj]] += counts[cond].to_list()
                    time_cond[potencias[jj]] = np.append(time_cond[potencias[jj]], time[cond])
                    cod_Ts[potencias[jj]] += cod[cond].to_list()
        for pot in potencias:
            counts_cond[pot] = np.array(counts_cond[pot])
        self.counts_cond = counts_cond
        self.time_cond = time_cond
        self.cod = cod_Ts
        self.potencias = potencias
        if save_data:
            np.save('time_cond_conrampas.npy', time_cond)
            np.save('counts_cond_conrampas.npy', counts_cond) 
            np.save('cod_Ts_conrampas.npy', cod_Ts)
            
    def apply_corr(self, ncond=2):
        time_norep = {}
        counts_norep = {}
        for pot in iter(self.potencias):
            cond = np.array([cod!=10 for cod in self.cod[pot]], dtype='bool')
            if cond.any() == True:
                t_rep, counts_rep = self.time_cond[pot][cond], self.counts_cond[pot][cond]
                time_norep[pot] = self.time_cond[pot][~cond]
                counts_norep[pot] = self.counts_cond[pot][~cond]
                t_rep_unique = pd.unique(t_rep)
                _, i_start, nrep = np.unique(t_rep, return_index=True, return_counts=True)
                res = np.split(counts_rep, i_start[1:])
                res_cond = [len(xx)>=ncond for xx in res]
                time_norep[pot] = np.append(time_norep[pot], t_rep_unique[res_cond])
                counts_norep[pot] = np.append(counts_norep[pot], np.array([np.mean(r)/0.1 for r in np.array(res, dtype='object')[res_cond]]))   
            else:
                time_norep[pot] = self.time_cond[pot]
                counts_norep[pot] = self.counts_cond[pot]
        self.time_norep  = time_norep
        self.counts_norep = counts_norep
    
    def moving_avg(self, N_vent = 10, N_sep =3):
        try:
            self.time_norep
            self.counts_norep
        except:
            self.apply_corr()
        time_deriva = {}
        counts_deriva = {}
        for pot in self.potencias:
            i_cut = np.array(np.where(np.diff(self.time_norep[pot])>=1/60*N_sep)[0] + 1, dtype = 'int64')
            t_cut, counts_cut = np.split(self.time_norep[pot], i_cut), np.split(self.counts_norep[pot], i_cut)
            counts_deriva[pot] = np.zeros((1, 2))
            time_deriva[pot] = np.zeros(1)
            for time, count in zip(t_cut, counts_cut):
                if len(count) > N_vent:
                    k = 0
                    while k < len(count) - N_vent:
                        counts_deriva[pot] = np.row_stack((counts_deriva[pot], np.array([np.mean(count[k:k+N_vent]), np.std(count[k:k+N_vent], ddof=1)])))
                        k+=1
                    time_deriva[pot] = np.append(time_deriva[pot], time[:-N_vent])
                elif len(count) == N_vent:
                    counts_deriva[pot] = np.row_stack((counts_deriva[pot], [np.mean(count), np.std(count, ddof=1)] ))
                    time_deriva[pot] = np.append(time_deriva[pot], time[0])
                else:
                    continue
            time_deriva[pot] = np.delete(time_deriva[pot], 0, axis=0)
            counts_deriva[pot] = np.delete(counts_deriva[pot], 0, axis=0)
        return time_deriva, counts_deriva

def average_stable(time, signal, step_times):
    potencias = signal.keys()
    counts_avg = np.zeros((len(potencias), 2))
    for jj, pot in enumerate(potencias):
        cond = np.logical_and(time[pot]>step_times[jj][0], time[pot]<step_times[jj][1])
        counts_avg[jj] = [np.mean(signal[pot][cond]), np.std(signal[pot][cond], ddof=1)]
    return counts_avg

#%% Figures 

def figure_SEAD(datos_sead, marchas_list, yscale = 'log', nfig = 99):
    fig = plt.figure(nfig)
    for marcha in marchas_list:
        plt.plot(datos_sead.time, datos_sead.CIC_Marcha(marcha), '.', label = marcha)
    plt.suptitle('Exported data from SEAD. Start time = {}'.format(datos_sead.t_inicio))
    plt.xlabel('t [min]')
    plt.ylabel('Current [A]')
    plt.legend()
    plt.grid(ls='--')
    plt.yscale(yscale)
    plt.tight_layout()  
    plt.show()
    return fig

def figure_LINT_err(LINT_t, LINT_counts, yscale = 'symlog', nfig = 98):
    fig = plt.figure(nfig)
    ax = fig.add_subplot()
    for t, count, pot in zip(LINT_t.values(), LINT_counts.values(), LINT_counts.keys()):
        ax.errorbar(t, count[:, 0], yerr= count[:, 1], fmt='.', label = pot + 'W')
    ax.set_xlabel('t [min]')
    ax.set_ylabel('CPS')
    ax.set_yscale(yscale)
    ax.legend()
    ax.grid(ls='--')
    fig.tight_layout()  
    fig.show()
    return fig, ax

def add_vlines(fig, ax, list_x, style = '--'):
    y0, y1 = ax.get_ylim()
    ax.vlines(list_x, ymin = y0, ymax = y1, ls = style, color = 'tab:gray')
    return fig

#%% Otras herramientas

def redondeo(mean, err, cs, texto = False):
    """
    Devuelve al valor medio con la misma cant. de decimales que el error (con 2 c.s.).
    """
    digits = -np.floor(np.log10(err)).astype(int)+cs-1
    if err<1:
        err_R = format(np.round(err, decimals = digits), f'.{digits}f')
        mean_R = str(np.round(mean, decimals = len(err_R)-2))
    else:
        err_R = format(np.round(err, decimals = digits), '.0f')
        mean_R = format(np.round(mean, decimals = cs-1-len(err_R)), '.0f')
    if texto == True:
        return (mean_R, '±',err_R)
    else:
        return (float(mean_R), float(err_R))

def ajuste_pol(grado, xdata, ydata, y_err):
    f = lambda x: np.column_stack([a*b for a, b in combinations(x, 2)])
    matrix_fromx = lambda data: np.column_stack([data**exp for exp in np.arange(grado+1)])
    cova_y = np.diag(y_err**2)
    design_matrix = matrix_fromx(xdata)
    cova_mle = np.linalg.inv(design_matrix.T @ np.linalg.inv(cova_y) @ design_matrix )
    matrix_B = cova_mle @ design_matrix.T @ np.linalg.inv(cova_y)
    par_est = matrix_B @ ydata
    par_err = np.sqrt(np.diag(cova_mle))
    comb = f(par_err) #combinatoria de stds de par_err
    upper_tri = cova_mle[np.triu_indices_from(cova_mle, k=1)] #upper triangle from covariance matrix
    rhos = upper_tri/comb #factores de correlación (01, 02, 03, ... 0max, 12, 13). Usa combinatoria de las stds y los elementos de la cov 
    var_mu = lambda xfit: matrix_fromx(xfit)**2 @ np.diag(cova_mle) + 2*np.row_stack([f(x) for x in matrix_fromx(xfit)]) @ upper_tri #varianza del parámetro hallado 
    
    residuos = ydata - design_matrix @ par_est
    J_min_observado = residuos.T @ np.linalg.inv(cova_y) @ residuos #chi2
    ddof = len(xdata) - len(par_est)
    pvalor = chi2.sf(J_min_observado, ddof) 
    return par_est[::-1], par_err[::-1], J_min_observado, residuos, pvalor, ddof, rhos, var_mu
