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

F_PotA = {'CIP': 3.006e11} #en W/A

class data_sead:
    def __init__(self, path_SEAD, fecha_ensayo = False, delim: str = ','):
        '''
        Inicia clase con datos del SEAD.

        Parameters
        ----------
        path_SEAD : str
            Ruta del archivo .csv con los datos exportados.
        fecha_ensayo : str (D = bool, False)
            Fecha del ensayo, con formato '05Jul2024'. Si no se elije una fecha de ensayo, se supone que no se necesita para leer 'Hora'.
        delim : str (D = ',')
            Delimitador del .csv. Suele ser ',' o ';'.
            
        Returns
        -------
        Agrega atribuciones:
            - data: datos importados
            - time: vector temporal en minutos, corregido a t_inicio
            - t_inicio: tiempo de inicio en el vector t crudo
            - fecha: fecha del ensayo
        '''
        data_sead = pd.read_csv(path_SEAD, delimiter = delim)
        if fecha_ensayo:
            data_sead['Hora'] = list(datetime.strptime(fecha_ensayo + '_'+ hora, '%d%b%Y_%H:%M:%S.%f') for hora in data_sead['Hora'])
            self.fecha = fecha_ensayo
        else:
            try:
                data_sead['Hora'] = list(datetime.strptime(hora, '%H:%M:%S') for hora in data_sead['Hora'])
            except:
                data_sead['Hora'] = list(datetime.strptime(hora, '%H:%M:%S.%f') for hora in data_sead['Hora'])
        t_inicio_sead = data_sead['Hora'][0]
        data_sead_time = np.array([(data_sead_time - t_inicio_sead).total_seconds() for data_sead_time in data_sead['Hora']])
        self.data = data_sead
        self.time = data_sead_time/60
        self.t_inicio = t_inicio_sead
    
    def copy(self):
        '''
        Genera una copia de la clase data_sead.

        Returns
        -------
        class
            Copia de data_sead con todos los atributos y datos generados hasta ese momento.

        '''
        return copy.deepcopy(self)
    
    def CIC_Marcha(self, det, idx: str='LOG'):
        '''
        Extrae de self.data la corriente o señal del detector seleccionado. Sirve más para CICs, Marchas, Arranques.

        Parameters
        ----------
        det : str
            Nombre del detector. Ej: 'M3', 'A1', 'BT3'.
        idx : str (D = 'LOG')
            Define la escala o id del detector. Ejemplos: 'LOG', 'LIN', 'MA', 'TSN', 'TASA'
        
        Returns
        -------
        current : Panda Series, float
            Columna de self.data correspondiente.

        '''
        current = self.data[f'{idx} {str(det)}'.strip()]
        return current
        
    def select_steps(self, det, detalle: str=''):
        '''
        Elegir intervalos estables a partir del gráfico.

        Parameters
        ----------
        det : list, str
            Señal a cargar usando self.CIC_Marcha.
        detalle : str, (D = '')
            Detalles para la selección de pasos. Sólo se usa en el nombre.
        Returns
        -------
        step_index : array, int
            Devuelve una lista con los índices de ti y tf para cada escalón.
        
        También se guarda el array en un .txt, bajo el nombre 'step_t_min_{self.fecha}_{detalle}.txt'
        '''
        t = self.time 
        i = self.CIC_Marcha(*det)
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
        try:
            np.savetxt(f'step_t_min_{self.fecha}_{detalle}.txt', step_index, delimiter='\t')
        except:
            np.savetxt(f'step_t_min_{detalle}.txt', step_index, delimiter='\t')
        plt.close(99)
        return step_index
    
    def per_step(self, det, path_step = None, power_list = list(range(50))):
        '''
        En un dict, guarda las partes de la señal por escalón/intervalo, seleccionado por select_steps.

        Parameters
        ----------
        det : str
            Nombre del detector cuya señal se va a levantar con CIC_Marcha.
        path_step : str (D = None)
            Ruta de ubicación del .txt con los índices de los pasos. Si se deja vacío, abre el método select_steps
        power_list : list (D = [0, 1, ..., 50]), str
            Lista de nombres o etiquetas de los pasos/intervalos, que serán las keys o llaves de los diccionarios finales. 
            Generalmente uso la potencia de cada escalón.

        Returns
        -------
        time_step : dict
            Devuelve un diccionario con el formato {potencia: array t del escalón}.
        data_step : dict
            Devuelve un diccionario con el formato {potencia: array señal (I) del escalón}.

        '''
        try:
            step_times = np.loadtxt(path_step)
        except:
            step_times = self.select_steps(det)
        data_step = {}
        time_step = {}

        for jj, interval in enumerate(step_times):
            cond_sead = np.logical_and(self.time>interval[0], self.time<interval[1])
            data_step[str(power_list[jj])] = self.CIC_Marcha(det)[cond_sead]
            time_step[str(power_list[jj])] = self.time[cond_sead]
        return time_step, data_step
    
    
def process_lint(path, timestamp: bool = True, pulse_py = False):
    '''
    Primer proceso de los datos crudos del LINT, adquiridos por graficarCampbell v1.0.0.0 (by Juan Alarcón)

    Parameters
    ----------
    path : str
        Define la ruta del archivo guardado por el LINT.
    timestamp : bool (D = True)
        Si False, significa que los datos se adquirieron con una versión del software que no tiene timestamp.
        Si True, el análisis tiene en cuenta las columnas del timestamp.
    Returns
    -------
    time (si timestamp = True) : array, float or datetime
        Vector de tiempo, con origen en t_inicio.
    counts : Series, int
        Cuentas en una unidad de sample time.
    cod : int
        Código del programa relacionado con el tiempo de sampleo (por ej., 1: 0.1s, 10: 1s).

    '''
    if pulse_py:
        data_lint = pd.read_table(path, sep = ",", engine = 'python')
        data_time = [datetime.strptime("_".join(hora), '%m-%d-%Y_%H:%M:%S.%f') for hora in data_lint.get(['Date', 'PC_Timestamp']).values]
        counts = data_lint['Counts']
        cod = pd.Series(np.ones(len(counts), dtype=int)*10)
        return data_time, counts, cod
    else:
        data_lint = pd.read_table(path, sep = ";|\\s+", header=None, engine = 'python')
        if timestamp:
            data_time = data_lint.get([x for x in range(6)])
            time = [datetime(*data_time.iloc[i].to_list()) for i in range(len(data_time))]
            counts = data_lint[6]
            cod = data_lint[7]
            return time, counts, cod        
        else:
            counts = data_lint[0]
            cod = data_lint[1]
            return counts, cod

class data_LINT:
    def __init__(self, path_LINT_list, path_step = False, t_inicio_SEAD = pd.Timestamp.now(), 
                 t_inicio = None, t_final = None, save_data=False, power_list = False, 
                 timestamp: bool = True, dt: float= 0, adqfrompy = False):
        '''
        Class para los datos del LINT. Al inicio, 

        Parameters
        ----------
        path_LINT_list : list, str
            Lista con las rutas de ubicación de los archivos .txt exportados del LINT.
        path_step : str, 
            Dirección en donde se encuentra el archivo con los ti, tf de los escalones.
        t_inicio : str (D = None)
            Tiempo inicial definido por el archivo del SEAD.
            Ejemplo: Timestamp('2024-07-05 11:20:00.453000')
        t_final : str (D = None)
            Sirve para definir 
        save_data : bpol (D = False)
            ¿Desea guardar los datos? Si es True, se guardan los arrays de counts, time, cod 
        power_list : TYPE, optional
            DESCRIPTION. The default is False.
        timestamp : bool (D = True)
            Si False, significa que los datos se adquirieron con una versión del software que no tiene timestamp.
            Si True, el análisis tiene en cuenta las columnas del timestamp.
        dt : float 
            Si hay un desfasaje temporal entre el SEAD y la PC del LINT, incluirlo acá: dt = tSEAD - tLINT
        
        Returns
        -------
        None.

        '''
        if power_list:
            potencias = power_list
            step_times = np.loadtxt(path_step)
        else:
            try:
                step_times = np.loadtxt(path_step) 
            except:
                step_times = [[0, 1e+20]] #default
            potencias = list(x for x in range(len(step_times)))
        counts_cond = {ii:[] for ii in potencias}
        time_cond = {ii:[] for ii in potencias}
        cod_Ts = {ii:[] for ii in potencias}
        time_cond_hhmmss = {ii:[] for ii in potencias}
        
        for path in path_LINT_list:
            if timestamp:
                time, counts, cod = process_lint(path, pulse_py = adqfrompy)
            else:
                counts, cod = process_lint(path, timestamp = timestamp)
                time = pd.date_range(t_inicio, t_final, periods = len(counts), tz=None)
            time_s = np.array([((tt - t_inicio_SEAD).total_seconds() + dt)/60 for tt in time])
            for jj, interval in enumerate(step_times):
                cond = np.logical_and(time_s>interval[0], time_s<interval[1])
                date = np.array([str(tt.time())[0:8] for tt in time])
                if any(cond) == True:    
                    counts_cond[potencias[jj]] += counts[cond].to_list()
                    time_cond[potencias[jj]] = np.append(time_cond[potencias[jj]], time_s[cond])
                    cod_Ts[potencias[jj]] += cod[cond].to_list()
                    time_cond_hhmmss[potencias[jj]] = np.append(time_cond_hhmmss[potencias[jj]], date[cond])
        for pot in potencias:
            counts_cond[pot] = np.array(counts_cond[pot])
        self.time_cond_hhmmss = time_cond_hhmmss        
        self.counts_cond = counts_cond
        self.time_cond = time_cond
        self.cod = cod_Ts
        self.potencias = potencias
        if save_data:
            np.save('time_cond_conrampas.npy', time_cond)
            np.save('counts_cond_conrampas.npy', counts_cond) 
            np.save('cod_Ts_conrampas.npy', cod_Ts)
            
    def corr_rep(self, ncond=2):
        '''
        Si se adquiere en períodos menores a 1 s, convierte los datos a CPS.

        Parameters
        ----------
        ncond : int (D = 2)
            Cantidad mínima de repeticiones dentro del segundo. 

        Returns
        -------
        Atributos counts_norep, time_norep: CPS y marcas de tiempo para cada segundo.

        '''
        time_norep = {}
        counts_norep = {}
        time_norep_hhmmss = {}
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
                try:
                    time_norep_hhmmss[pot] = self.time_cond_hhmmss[pot][~cond]
                    time_norep_hhmmss[pot] = np.append(time_norep_hhmmss[pot], pd.unique(self.time_cond_hhmmss[pot][cond])[res_cond])
                except:
                    continue
            else:
                time_norep_hhmmss[pot] = self.time_cond[pot]
                time_norep[pot] = self.time_cond[pot]
                counts_norep[pot] = self.counts_cond[pot]
        self.time_norep  = time_norep
        self.counts_norep = counts_norep
        if len(time_norep_hhmmss)!=0:
            self.time_norep_hhmmss = time_norep_hhmmss
    def corr_nonpar(self, tdead):
        '''
        Non-paralizable detector correction.

        Parameters
        ----------
        tdead : float
            Tiempo muerto [s] característico del detector.

        Returns
        -------
        Attribute 'counts_corr', dict

        '''
        m_corr = {}
        try:
            m = self.counts_norep
        except:
            self.corr_rep()
            m = self.counts_norep
        for pot in self.potencias:
            m[pot] = np.array(m[pot])
            m_corr[pot] = m[pot]/(1-m[pot]*tdead)
        self.counts_corr = m_corr
  
    def stable_interval(self, path_stable):
        stable_times = np.loadtxt(path_stable)
        potencias = self.potencias
        times_stable = {}
        counts_stable = {}
        for jj, pot in enumerate(potencias):
            cond = np.logical_and(self.time_norep[pot]>stable_times[jj][0], self.time_norep[pot]<stable_times[jj][1])
            counts_stable[pot] = self.counts_norep[pot][cond]
            times_stable[pot] = self.time_norep[pot][cond]
        self.counts_stable = counts_stable
        self.time_stable = times_stable


def moving_avg(x, y, N_vent = 10, N_sep =3, is_mean: bool=True):
    '''
    Calcula el promedio móvil de algún set 'y' a partir de su vector temporal 'x', para cada N_vent cantidad de adquisiciones.
    
    
    Parameters
    ----------
    x : array
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    N_vent : TYPE, optional
        DESCRIPTION. The default is 10.
    N_sep : TYPE, optional
        DESCRIPTION. The default is 3.
    is_mean : bool, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    x_deriva : TYPE
        DESCRIPTION.
    y_deriva : TYPE
        DESCRIPTION.

    '''
    if type(y)!= dict:
        y = {'0': y}
    if type(x)!=dict:
        x = {'0': x}        
    x_deriva = {}
    y_deriva = {}
    for pot in y:
        i_cut = np.array(np.where(np.diff(x[pot])>=1/60*N_sep)[0] + 1, dtype = 'int64')
        x_cut, y_cut = np.split(x[pot], i_cut), np.split(y[pot], i_cut)
        y_deriva[pot] = np.zeros((1, 2))
        x_deriva[pot] = np.zeros(1)
        if is_mean:
            f = lambda x: np.mean(x)
        else:
            f = lambda x: np.sum(x)
        for time, count in zip(x_cut, y_cut):
            if len(count) > N_vent:
                k = 0
                while k < len(count) - N_vent:
                    y_deriva[pot] = np.row_stack((y_deriva[pot], np.array([f(count[k:k+N_vent]), np.std(count[k:k+N_vent], ddof=1)])))
                    k+=1
                x_deriva[pot] = np.append(x_deriva[pot], time[:-N_vent])
            elif len(count) == N_vent:
                y_deriva[pot] = np.row_stack((y_deriva[pot], [f(count), np.std(count, ddof=1)] ))
                x_deriva[pot] = np.append(x_deriva[pot], time[0])
            else:
                continue
        x_deriva[pot] = np.delete(x_deriva[pot], 0, axis=0)
        y_deriva[pot] = np.delete(y_deriva[pot], 0, axis=0)
    return x_deriva, y_deriva   
        
def average_stable(time, signal, step_times): #agregar intervalos estables a data_LINT, para hacer análisis sobre ellos.
    potencias = signal.keys()
    counts_avg = np.zeros((len(potencias), 2))
    N_avg = np.zeros(len(potencias))
    for jj, pot in enumerate(potencias):
        cond = np.logical_and(time[pot]>step_times[jj][0], time[pot]<step_times[jj][1])
        N_avg[jj] = len(signal[pot][cond])
        counts_avg[jj] = [np.mean(signal[pot][cond]), np.std(signal[pot][cond], ddof=1)/np.sqrt(N_avg[jj])]
    return counts_avg, N_avg

def sel_data(datalist, index):
    i, f = index
    if i<0:
        f, i = abs(i), abs(f+1)
        for ii, data in enumerate(datalist):
            datalist[ii] = data[::-1][i:f][::-1]
    else:
        for ii, data in enumerate(datalist):
            datalist[ii] = data[i:f]
    return datalist    

def t_comparison(tcomp, tref, tolerance: float=0.5):
    #desde dicts, con keys para cada potencia
    indcomp_sel = []
    indref_sel = []
    for jj, t2 in enumerate(tref):
        x = (tcomp - t2)*60
        i_sel = np.where(abs(x)<=tolerance)[0]
        if len(i_sel)!=0:
            indref_sel.append(jj)
            indcomp_sel.append(i_sel[0])
    return indcomp_sel, indref_sel
        
def select_data(measured, data_table, tolerance, idx=0): 
    rows, cols = len(measured), len(data_table.T)
    data_sel = np.zeros((rows, cols-1))
    for jj, en in enumerate(measured):
        for ee in data_table:
            x = en/ee[idx]
            if 1-tolerance<x<1+tolerance:
                data_sel[jj] = np.delete(ee, idx)
    return data_sel
    

#%% Figures 

def LINT_hist_step(self, pot_sel):
    """
    Gráfico de histograma por paso, a partir de datos corregidos

    Parameters
    ----------
    pot_sel : list, str
        Potencia seleccionada para hacer el histograma.

    Returns
    -------
    Shows figures.

    """
    for pot in pot_sel:
        res = self.counts_norep[pot]
        plt.figure()
        plt.hist(res, bins=int(len(res)**(1/3)), label = pot + 'W')
        plt.xlabel('Cuentas')
        plt.ylabel('Frecuencia')
        plt.legend()
        plt.tight_layout()

def figure_SEAD(datos_sead, marchas_list, idx_list, yscale = 'log', nfig = 99):
    fig, ax = plt.subplots()
    for marcha, idx in zip(marchas_list, idx_list):
        ax.plot(datos_sead.time, datos_sead.CIC_Marcha(marcha, idx = idx), '.', label = (idx+' ' + marcha).strip())
    fig.suptitle('Exported data from SEAD. Start time = {}'.format(datos_sead.t_inicio))
    ax.set_xlabel('t [min]', fontsize = 12)
    ax.set_ylabel('Current [A]', fontsize = 12)
    ax.legend(fontsize = 12)
    ax.grid(ls='--')
    ax.set_yscale(yscale)
    fig.tight_layout()  
    fig.show()
    return fig, ax

def fig_LINT_MX_tuple(LINT_counts, LINT_time, Current_M, Time_M, yscale = 'linear', power_sel = [], nfig = 98):
    if power_sel != []:
        Current_M = {x: Current_M[x] for x in power_sel}
        LINT_counts = {x: LINT_counts[x] for x in power_sel}
        LINT_time = {x: LINT_time[x] for x in power_sel}
        Time_M = {x: Time_M[x] for x in power_sel}
    fig = plt.figure(nfig)
    # plt.clf()
    # i_sels = {}
    totalcount_sel = np.array([])
    totalcurrent_sel = np.array([])
    totaltime_sel = np.array([])
    for count, current, pot, tlint, tmarcha in zip(LINT_counts.values(), Current_M.values(), list(LINT_counts.keys()),
                                                   LINT_time.values(), Time_M.values()):
        icomp, iref = t_comparison(tlint, tmarcha)
        # i_sels[pot] = i_sel  
        plt.plot(current.to_numpy()[iref], count[icomp], '.', label = pot + 'W')
        totalcurrent_sel = np.append(totalcurrent_sel, current.to_numpy()[iref])
        totalcount_sel = np.append(totalcount_sel, count[icomp])
        totaltime_sel = np.append(totaltime_sel, tmarcha[iref])
    plt.ylabel('CPS')
    plt.xlabel('Current [A]')
    plt.yscale(yscale)
    plt.legend()
    plt.grid(ls='--')
    fig.tight_layout()  
    # fig.show()
    return fig, totalcount_sel, totalcurrent_sel, totaltime_sel
    
def figure_LINT_err(x_data, y_data, yscale = 'symlog', power_sel = ['0'], nfig = 98, 
                    labels = ['t [min]', 'Tasa [cps]'], error: bool=True):
    if power_sel != ['0']:
        LINT_t = {x: x_data[x] for x in power_sel}
        LINT_counts = {x: y_data[x] for x in power_sel}
    fig = plt.figure(nfig)
    ax = fig.add_subplot()
    if error:
        for t, count, pot in zip(LINT_t.values(), LINT_counts.values(), LINT_counts.keys()):
            ax.errorbar(t, count[:, 0], yerr= count[:, 1], ecolor= 'tab:gray', fmt='r.', label = str(pot) + 'W')
    else:
        for t, count, pot in zip(LINT_t.values(), LINT_counts.values(), LINT_counts.keys()):
            ax.plot(t, count[:, 0], 'r.', label = str(pot) + 'W')
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_yscale(yscale)
    # ax.legend()
    ax.grid(ls='--')
    fig.tight_layout()  
    fig.show()
    return fig, ax

def add_vlines(fig, ax, list_x, style = '--'):
    y0, y1 = ax.get_ylim()
    ax.vlines(list_x, ymin = y0, ymax = y1, ls = style, color = 'tab:gray')
    x_paired = np.split(list_x, len(list_x)//2)
    for x in x_paired:
        ax.fill_between(x, y0, y1, alpha=0.5, color='tab:olive')
    return fig

def figure_fit(xdata, ydata, yerror, fit_data, scale = 'log', i_sel = [], xerror = None, show_report = True):
    if i_sel != []:
        xdata, ydata, yerror = sel_data([xdata, ydata, yerror], i_sel)
    try:
        xerror = xerror[i_sel[0]:i_sel[1]]
    except:
        xerror = np.zeros(len(xdata))
    x = np.linspace(np.min(xdata), np.max(xdata), 1000)
    sigma_mu_est = np.sqrt(fit_data[-1](x))
    yerr_eff = ydata*np.sqrt((yerror/ydata)**2 + (xerror/xdata)**2)
    y_eval = np.polyval(fit_data[0], x)
    
    plt.figure()
    plt.errorbar(xdata, ydata, yerr = yerr_eff, fmt = '.', label = 'datos')
    plt.plot(x, y_eval, label = 'Ajuste', linewidth = 2.0)
    plt.fill_between(x, y_eval-sigma_mu_est, y_eval+sigma_mu_est, color='tab:orange', alpha=0.2)
    plt.legend()
    plt.grid(ls = '--')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.xlabel('Corriente [A]')
    plt.ylabel('Tasa de conteo [CPS]')
    plt.xscale(scale)
    plt.yscale(scale)
    plt.tight_layout()
    
    if show_report:
        report = 'Fit Report: \n'
        for var, error, par in zip(fit_data[0], fit_data[1], ['a', 'b']):
            report += par + " = " + ''.join(redondeo(var, error, 3, texto= True)) + ' \n'
        report += "p-valor = " + str(fit_data[4]) + '\n'
        report += "chi2 = " + str(fit_data[2]) + '\n'
        report += "dof = " + str(fit_data[5]) + '\n'
        print(report)
    plt.show()

def fig_err(counts_std, counts, power_list, scale = 'log'):
    plt.figure()
    for count, count_std, power in zip(counts, counts_std, power_list):
        plt.plot(count_std, np.sqrt(count), 'o', label = f'{power}W')
    plt.legend()
    plt.ylabel(r'$\sqrt{Counts}$')
    plt.xlabel(r'$\sigma_{std}$')
    plt.grid(ls = '--')
    plt.xscale(scale)
    plt.yscale(scale)
    plt.tight_layout()

def fig_LINT_attr(data_LINT, data_sel: list):    
    x, y = [list(getattr(data_LINT, name).values()) for name in data_sel]
    fig, ax = plt.subplots()
    ax.plot(np.concatenate(x), np.concatenate(y), '.')
    ax.grid(ls = '--')
    return fig, ax

#%% Otras herramientas

def redondeo(mean, err, cs, texto = False):
    """
    Devuelve al valor medio con la misma cant. de decimales que el error (con 2 c.s.).
    """
    if np.log10(abs(mean/err)) <= 1E7:
        digits = -np.floor(np.log10(err)).astype(int)+cs-1
        if err<1:
            digits +=1
            err_R = format(np.round(err, decimals = digits), f'.{digits}f')
            mean_R = str(np.round(mean, decimals = len(err_R)-2))
        else:
            if digits<=0:    
                err_R = format(np.round(err, decimals = digits), '.0f')
            else:
                err_R = format(np.round(err, decimals = digits))
            mean_R = format(np.round(mean, decimals = cs-1-len(err_R)), '.0f')
    else:
        err_R = 0
        mean_R = mean
    if texto == True:
        mean_R, err_R = float(mean_R), float(err_R)
        if err_R<1 or mean_R<1:
            q = int(max(np.log10(abs(mean_R)), np.log10(err_R))-1)
            return (str(mean_R*10**-q), '±', '{:f}'.format(err_R*10**-q), f' · 1E{q}')
        else:
            return (str(mean_R), '±', str(err_R))
    else:
        return (mean_R, err_R)

def ajuste_pol(grado, xdata, ydata, y_err, i_sel=[]):
    if i_sel != []:
        xdata, ydata, y_err = sel_data([xdata, ydata, y_err], i_sel)
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


