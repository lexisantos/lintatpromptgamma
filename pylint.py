import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime
import copy

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
    
    def per_step(self, det, path_step = None):
        try:
            step_times = np.loadtxt(path_step)
        except:
            step_times = self.select_steps(self, det)
        data_step = {}
        time_step = {}

        for jj, interval in enumerate(step_times):
            cond_sead = np.logical_and(self.time>interval[0], self.time<interval[1])
            data_step[jj] = self.CIC_Marcha(det)[cond_sead]
            time_step[jj] = self.time[cond_sead]
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
    def __init__(self, path_LINT_list, path_step, t_inicio, save_data=False):
        step_times = np.loadtxt(path_step)
        counts_cond = {ii:[] for ii in range(len(step_times))}
        time_cond = {ii:[] for ii in range(len(step_times))}
        cod_Ts = {ii:[] for ii in range(len(step_times))}
        for path in path_LINT_list:
            time, counts, cod = process_lint(path, t_inicio)
            for jj, interval in enumerate(step_times):
                cond = np.logical_and(time>interval[0], time<interval[1])
                if any(cond) == True:    
                    counts_cond[jj] += counts[cond].to_list()
                    time_cond[jj] = np.append(time_cond[jj], time[cond])
                    cod_Ts[jj] += cod[cond].to_list()
        for jj in counts_cond:
            counts_cond[jj] = np.array(counts_cond[jj])
        self.counts_cond = counts_cond
        self.time_cond = time_cond
        self.cod = cod_Ts
        if save_data:
            np.save('time_cond_conrampas.npy', time_cond)
            np.save('counts_cond_conrampas.npy', counts_cond) 
            np.save('cod_Ts_conrampas.npy', cod_Ts)

    def copy(self):
        return copy.deepcopy(self)

    def apply_corr(self, ncond=2):
        time_norep = {}
        counts_norep = {}
        for i in self.time_cond:
            cond = np.array([cod!=10 for cod in self.cod[i]])
            if cond.any() == True:
                t_rep, counts_rep = self.time_cond[i][cond], self.counts_cond[i][cond]
                time_norep[i] = self.time_cond[i][~cond]
                counts_norep[i] = self.counts_cond[i][~cond]
                t_rep_unique = pd.unique(t_rep)
                _, i_start, nrep = np.unique(t_rep, return_index=True, return_counts=True)
                res = np.split(counts_rep, i_start[1:])
                res_cond = [len(xx)>=ncond for xx in res]
                time_norep[i] = np.append(time_norep[i], t_rep_unique[res_cond])
                counts_norep[i] = np.append(counts_norep[i], np.array([np.mean(r)/0.1 for r in np.array(res, dtype='object')[res_cond]]))   
            else:
                time_norep[i] = self.time_cond[i]
                counts_norep[i] = self.counts_cond[i]
        self.time_norep  = time_norep
        self.counts_norep = counts_norep
    
    def moving_avg(self, N_vent = 10, N_sep = 3):
        try:
            self.time_norep
            self.counts_norep
        except:
            self.apply_corr()
        time_deriva = {}
        counts_deriva = {}
        for i in self.time_norep:
            i_cut = np.array(np.where(np.diff(self.time_norep[i])>=1/60*N_sep)[0] + 1, dtype = 'int64')
            t_cut, counts_cut = np.split(self.time_norep[i], i_cut), np.split(self.counts_norep[i], i_cut)
            counts_deriva[i] = np.zeros((1, 2))
            time_deriva[i] = np.zeros(1)
            for time, count in zip(t_cut, counts_cut):
                if len(count) > N_vent:
                    k = 0
                    while k < len(count) - N_vent:
                        counts_deriva[i] = np.row_stack((counts_deriva[i], np.array([np.mean(count[k:k+N_vent]), np.std(count[k:k+N_vent], ddof=1)])))
                        k+=1
                    time_deriva[i] = np.append(time_deriva[i], time[:-N_vent])
                elif len(count) == N_vent:
                    counts_deriva[i] = np.row_stack((counts_deriva[i], [np.mean(count), np.std(count, ddof=1)] ))
                    time_deriva[i] = np.append(time_deriva[i], time[0])
                else:
                    continue
            time_deriva[i] = np.delete(time_deriva[i], 0, axis=0)
            counts_deriva[i] = np.delete(counts_deriva[i], 0, axis=0)
        return time_deriva, counts_deriva



#%% Figures 

def figure_SEAD(datos_sead, marchas_list, yscale = 'log', nfig = 1):
    fig = plt.figure(nfig)
    for marcha in marchas_list:
        plt.plot(datos_sead.time, datos_sead.CIC_Marcha(marcha), '.', label = marcha)
    plt.xlabel('t [min]')
    plt.ylabel('Current [A]')
    plt.legend()
    plt.grid(ls='--')
    plt.yscale(yscale)
    plt.tight_layout()  
    plt.close()
    return fig

def figure_LINT_err(LINT_t, LINT_counts, yscale = 'symlog',
                    power_list = [], nfig = 98):
    fig = plt.figure(nfig)
    axs = []
    for t, count in zip(LINT_t.values(), LINT_counts.values()):
        ax = plt.errorbar(t, count[:, 0], yerr= count[:, 1], fmt='.')
        axs.append(ax)
    plt.xlabel('t [min]')
    plt.ylabel('CPS')
    plt.yscale(yscale)
    if power_list!=[]:
        plt.legend(axs, power_list)
    plt.grid(ls='--')
    plt.tight_layout()  
    plt.close()
    return fig

