# LINTatPromptGamma

`pylint.py` contiene funciones útiles para analizar y graficar los datos devueltos por el SEAD y el software de adquisición del LINT. Se irá actualizando a medida que se necesiten nuevos métodos de análisis.

`LINT_pulsecounter.py` permite comunicarse con el contador de pulsos, grabar [fecha, hora PC, t [s], cuentas] y graficar en simultáneo durante la adquisición de datos. Se pudo testear el funcionamiento con y sin un input en el contador.

## Acerca de
Estos scripts son usados bajo el marco del proyecto LINT, cuyo objetivo es instalar y caracterizar un detector de neutrones térmicos en la columna del Prompt Gamma del RA-3. Los datos y resultados (si se encuentra visible, en `AnalisisLINT.ipynb`) se usarán para el desarrollo de mi tesis de Lic. en Física. 