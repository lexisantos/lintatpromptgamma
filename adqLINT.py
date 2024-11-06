# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 07:40:19 2024

@author: Alex
"""

# import glob
from Codigos_py.Repositorio import pylint
from pylint import np, plt, pd

path_SEAD =  ""
path_LINT_list = ['D:/Proyecto LINT at Prompt Gamma/2024-07-05_experimentAtPromptGamma/datos_05.07.24/datos_contador_LINT-1/2024-07-05_15-04_reactorON_1MW_pos5.log']#glob.glob(r"*.log")
data_sel = ['counts_corr', 'time_norep', 'cod'] #['counts_stable', 'time_stable', 'cod'] #

t_inicio = pd.Timestamp('2024-11-06 11:00:00.000000') #Set initial time


data_LINT = pylint.data_LINT(path_LINT_list)
data_LINT.corr_rep()
data_LINT.corr_nonpar(200e-9)

N_t = 10

cps, t, cod = [list(getattr(data_LINT, name).values())[0] for name in data_sel]

fig, ax = plt.subplots()
ax.plot(cps, '.')
t_fix = pd.date_range(f'{t[0]}', f'{t[-1]}', periods = N_t)
ax.set_xticks(ticks = np.linspace(0, len(cps), num=N_t), labels = [str(tt)[0:8] for tt in t_fix.time])
fig.autofmt_xdate()
