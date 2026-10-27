import numpy as np

massa_stick = 0.560  # kg
error_massa = 1e-3 # kg


error_lengte_metingen = 0.5e-3 # 0.5mm

# staaf (cilinder), in meter
diameter_stick = 3.15e-2
straal_stick = diameter_stick / 2
error_straal_stick = error_lengte_metingen / 2
lengte_staaf = 85.75e-2

# blok, in meter
hoogte_blok = 15e-3      # afmeting langs de bewegingsrichting
breedte_blok = 7.6e-2          # afmeting langs de staaf
lengte_blok = 11.3e-2         # afmeting langs de draaias van de hockeystick

volume_staaf = np.pi * straal_stick**2 * lengte_staaf
error_volume_staaf =  volume_staaf * np.sqrt((2*error_straal_stick/straal_stick)**2 + (error_lengte_metingen/lengte_staaf)**2)
volume_blok = hoogte_blok * breedte_blok * lengte_blok
error_volume_blok = volume_blok * np.sqrt((error_lengte_metingen/hoogte_blok)**2 + (error_lengte_metingen/breedte_blok)**2 + (error_lengte_metingen/lengte_blok)**2)


dichtheid = massa_stick / (volume_staaf + volume_blok)
error_dichtheid = dichtheid * np.sqrt((error_massa/massa_stick)**2 + (error_volume_blok**2 + error_volume_staaf**2) / (volume_blok + volume_staaf)**2)
massa_staaf = dichtheid * volume_staaf
error_massa_staaf = massa_staaf * np.sqrt((error_dichtheid/dichtheid)**2 + (error_volume_staaf/volume_staaf)**2)
massa_blok = dichtheid * volume_blok
error_massa_blok = massa_blok * np.sqrt((error_dichtheid/dichtheid)**2 + (error_volume_blok/volume_blok)**2)

# afstand van het massamiddelpunt tot de draaias
d_staaf = lengte_staaf / 2
error_d_staaf = error_lengte_metingen / 2
d_blok = lengte_staaf + breedte_blok / 2
error_d_blok =  error_lengte_metingen / np.sqrt(2)

# traagheidsmomenten van elk onderdeel
I_staaf = (1/3) * massa_staaf * lengte_staaf**2
error_i_staaf = I_staaf * np.sqrt((error_massa_staaf/massa_staaf)**2 + (2*error_lengte_metingen/lengte_staaf)**2)
I_blok = (1/12)*massa_blok*(breedte_blok**2 + hoogte_blok**2) + massa_blok*d_blok**2
error_i_blok = (1/12)*massa_blok*(breedte_blok**2 + hoogte_blok**2) * np.sqrt((error_massa_blok/massa_blok)**2 + (2*error_lengte_metingen)**2*(breedte_blok**2 + hoogte_blok**2)) +  massa_blok*d_blok**2 * np.sqrt((error_massa_blok/massa_blok)**2 + (2*error_d_blok/d_blok)**2)
I_totaal = I_staaf + I_blok
error_totaal = np.sqrt(error_i_staaf**2 + error_i_blok**2)


if __name__ == "__main__":
    print(I_totaal)
    print(error_totaal)
