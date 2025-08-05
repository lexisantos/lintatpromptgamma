# LINTatPromptGamma
Breve descripción de los scripts y notebooks:

- `pylint.py`:
Contiene funciones útiles para analizar y graficar los datos devueltos por el SEAD y el software de adquisición del LINT. 
_It defines methods and classes, useful for processing, analizing and plotting the data (-.csv) from the Electronic Acquisition Data System at the RA-3 and the output (-.txt) from the Li detector._

- `LINT_pulsecounter.py`: 
Permite comunicarse con el contador de pulsos, grabar [fecha, hora PC, t [s], cuentas] y graficar en simultáneo durante la adquisición de datos. Se pudo testear el funcionamiento con y sin un input en el contador.
_It allows measurement automation. Reads the terminal from the Pulse Counter and writes on a new .txt file, adding the date and a timestamp: [Date, Time PC, t [s], Counts]. Simultaneously, plots the data Counts vs t [s]._

- `HPGe_PGflux.ipynb`: 
Muestra el análisis y los resultados de las mediciones de flujo en la columna del Prompt Gamma. Hecho originalmente en Google Colab para poder compartirlo. Aprovecha métodos ya definidos en Activación.py (from [HPGe_Spectrum](https://gitlab.com/alexcanchanya/hpge-spectrum)).
_PNGAA analysis using metallic foils. It involved the calibration of two HPGe detectors at different sets, and the measurement of thermal flux at different position in and around the neutron beam. Originally created in Google Colab. Methods from HPGe Spectrum/Activacion.py are used._

- `AnalisisLINT.ipynb`:
Resultados del ensayo de linealidad en el haz del PG. Se hizo una calibración cruzada con otros detectores, aunque las constantes no sean de utilidad. Se verifica linealidad a partir de la bondad del ajuste. Se observa buen seguimiento de las tendencias ante cambios de flujo.
_Linearity study at the PG Facility Beam. Cross calibration made in comparison to other detectors from the reactor. Although the parameters returned by the model fit are not important, we rely on the goodness of fitness to verify linearity. It was observed that the LINT is able to follow the same tendencies as the other detectors -or in other words, the flux changes._

## Acerca de
Estos scripts son usados bajo el marco del proyecto LINT, cuyo objetivo es instalar y caracterizar un detector de neutrones térmicos en la columna del Prompt Gamma del RA-3. Los resultados (en .ipynb) serán publicados en un artículo (próximamente). 
_These scripts are part of the LINT Project, a collaboration with the RA-3. The main purpose is to tune-up and install a thermal neutron detector at the Prompt Gamma Facility. The results (.ipynb) will be published in an article (coming soon)._
