# LINTatPromptGamma

- `pylint.py`:
Contiene funciones útiles para analizar y graficar los datos devueltos por el SEAD y el software de adquisición del LINT. 
_It defines methods and classes, useful for processing, analizing and plotting the data (-.csv) from the Electronic Acquisition Data System at the RA-3 and the output (-.txt) from the Li detector._

`LINT_pulsecounter.py`: 
    Permite comunicarse con el contador de pulsos, grabar [fecha, hora PC, t [s], cuentas] y graficar en simultáneo durante la adquisición de datos. Se pudo testear el funcionamiento con y sin un input en el contador.
    It allows measurement automation. Reads the terminal from the Pulse Counter and writes on a new .txt file, adding the date and a timestamp: [Date, Time PC, t [s], Counts]. Simultaneously, plots the data Counts vs t [s]. 

`HPGe_PGflux` contiene análisis y resultados de las mediciones de flujo en la columna del Prompt Gamma. Hecho originalmente en Google Colab para poder compartirlo.

`LINT_pulsecounter.py` permite comunicarse con el contador de pulsos, grabar [fecha, hora PC, t [s], cuentas] y graficar en simultáneo durante la adquisición de datos. Se pudo testear el funcionamiento con y sin un input en el contador.


## Acerca de
Estos scripts son usados bajo el marco del proyecto LINT, cuyo objetivo es instalar y caracterizar un detector de neutrones térmicos en la columna del Prompt Gamma del RA-3. Los datos y resultados (si se encuentra visible, en `AnalisisLINT.ipynb`) se usarán para el desarrollo de mi tesis de Lic. en Física. 