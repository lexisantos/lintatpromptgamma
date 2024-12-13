# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 10:46:16 2024

@author: Alex
"""
import serial, datetime, csv, os, time
import numpy as np
import serial.tools.list_ports as stl
import matplotlib.pyplot as plt
import pyvisa as visa

def Timestamp():
    return datetime.datetime.now().strftime('%m-%d-%Y %H:%M:%S.%f')

def Show_ports():
    ports = stl.comports()
    for port, desc, hwid in sorted(ports):
        #puerto, descripcion, descripción técnica
        print("{}: {} [{}]".format(port, desc, hwid))
            
class PulseCounter:
    def __init__(self, port, baudrate = 57600):
        '''
        Parameters
        ----------
        port : str
            Indique el puerto al que está asociado.
        baudrate : int
            Number of signal changes, or symbols, that pass through a transmission 
            medium per second. (D = 57600).

        Returns
        -------
        None.

        '''
        global ser
        ser = serial.Serial() 
        ser.baudrate = baudrate
        ser.port = port
        
    def Initialize(self):
        '''
        Abrir el puerto serie.

        Returns
        -------
        None.

        '''
        global ser
        try:
            if ser.isOpen():
                print(f"\n Serial in {ser.port} is already Open. ")
            else:
                print("\n Error: Serial is not Open. Opening...")
                ser.open()
                print(f"\n Serial in {ser.port} is now Open.")
        except:
            print('\n Please, verify connection.')
    
    def Reader(self, filename = 'test_csv.csv', timeout: int=20, header = ['Date', 'PC_Timestamp', 't_LINT [s]', 'Counts'], visual = 'plot'):
        '''
        Lector de líneas del contador. Guarda y grafica/imprime a la vez.

        Parameters
        ----------
        filename : TYPE, optional
            Nombre del archivo al cual va a guardarse. The default is 'test_csv.csv'.
        timeout : int, optional
            DESCRIPTION. The default is 20.
        header : TYPE, optional
            DESCRIPTION. The default is ['Date', 'PC_Timestamp', 't_LINT [s]', 'Counts'].
        visual : TYPE, optional
            DESCRIPTION. The default is 'plot'.

        Returns
        -------
        None.

        '''
        read_arr = np.zeros((timeout, 2), dtype=int)
        with open(filename, 'w', newline='') as f:
            csv_file = csv.writer(f)
            csv_file.writerow(header)
            # ser.reset_input_buffer()
            read_arr[0] = np.array(ser.readline().decode("utf-8").split(';'), dtype=int) 
            t0 = read_arr[0, 0]
            read_arr[0, 0] = 0
            linecsv = [*Timestamp().split(), *read_arr[0]]
            csv_file.writerow(linecsv)
            n = 1
            while n < timeout:
                read_arr[n] = np.array(ser.readline().decode("utf-8").split(';'), dtype=int)
                read_arr[n, 0] -= t0
                linecsv = [*Timestamp().split(), *read_arr[n]]
                csv_file.writerow(linecsv)
                if visual.lower() == 'print':
                    print(*linecsv)
                elif visual.lower() == 'plot':
                    line1.set_xdata(read_arr[:n+1, 0])
                    line1.set_ydata(read_arr[:n+1, 1])
                    ax.set_xlim(read_arr[:n+1, 0].min(), read_arr[:n+1, 0].max())
                    ax.set_ylim(read_arr[:n+1, 1].min(), read_arr[:n+1, 1].max())
                    fig.canvas.draw()
                    fig.canvas.flush_events()
                else:
                    continue
                n+=1
            f.close()
        ser.reset_output_buffer()
        ser.close()
        print(f'\n Data saved in {os.getcwd()}{filename}')

class FG_Tektronix:
    def __init__(self, name = 'USB0::0x0699::0x0346::C035965::INSTR'):
        global afg
        rm = visa.ResourceManager()
        if name in (rm.list_resources()):
            print('AFG found. Connecting...')
            afg = rm.open_resource(name) 
            print('Connected to', afg.query('*IDN?'))
        else:
            print('AFG not found. Retry again')
        afg.write('OUTP1:STAT 1')
    def set_freq(self, freq):
        afg.write(f'SOUR1:FREQ {freq}')
    def set_amp(self, amp):
        afg.write(f'SOUR1:VOLT:AMPL {amp}')
    def set_function(self, form = 'PULS'):
        '''
        Available forms: SINusoid|SQUare|PULSe|RAMP|PRNoise|DC|SINC|GAUSsian|
LORentz|ERISe|EDECay|HAVersine

        Parameters
        ----------
        form : str, optional
            DESCRIPTION. The default is 'PULS'.

        Returns
        -------
        None.

        '''
        afg.write(f'SOUR1:FUNC:SHAP {form}')
    def set_width(self, width, unit = 'ns'):
        afg.write(f'SOUR1:PULS:WIDT {width}{unit}')
    def close_ch(self):
        afg.write('OUTP1:STAT 0')

def run(title, N, puerto = 'COM3', visual = 'plot'):
    global cp
    if visual.lower() == 'plot':
        global ax, fig, line1
        plt.ion()
        fig, ax = plt.subplots(figsize = (8,6))
        ax.set_xlabel('t [s]')
        ax.set_ylabel('Counts')
        ax.grid(True, ls='--')
        ax.ticklabel_format(useOffset = False)
    
        line1, = ax.plot([], [], "r.-")
    
    cp = PulseCounter(puerto) 
    cp.Initialize()
    cp.Reader(filename = title, timeout = N, visual = visual)


Show_ports()
#El LINT es USB VID:PID=0403:6015 SER=FTXW3XZ4A
#El AFG3021B es USB0::0x0699::0x0346::C035965::INSTR | SER = C035965

# %% 
freqs = 7*np.logspace(1, 4, num=20, endpoint = True, dtype=int)
f_afg = FG_Tektronix()
f_afg.set_amp(5)
f_afg.set_function('PULS')
f_afg.set_width(500)

for freq in freqs:
    f_afg.set_freq(freq)
    time.sleep(300)

f_afg.close_ch()

#%%
run('BarridoconGF_50a1kHz.csv', N = 6020, visual = 'plot')
