import numpy as np

massa_stick = 0.524  # kg

# staaf (cilinder), in meter
diameter_stick = 3.15e-2
straal_stick = diameter_stick / 2
lengte_staaf = 85.75e-2

# blok, in meter
hoogte_blok = 15e-3      # afmeting langs de bewegingsrichting
breedte_blok = 7.6e-2          # afmeting langs de staaf
lengte_blok = 11.3e-2         # afmeting langs de draaias van de hockeystick

volume_staaf = np.pi * straal_stick**2 * lengte_staaf
volume_blok = hoogte_blok * breedte_blok * lengte_blok

dichtheid = massa_stick / (volume_staaf + volume_blok)
massa_staaf = dichtheid * volume_staaf
massa_blok = dichtheid * volume_blok

# afstand van het massamiddelpunt tot de draaias
d_staaf = lengte_staaf / 2
d_blok = lengte_staaf + breedte_blok / 2

# traagheidsmomenten van elk onderdeel
I_staaf = (1/3) * massa_staaf * lengte_staaf**2
I_blok = (1/12)*massa_blok*(breedte_blok**2 + hoogte_blok**2) + massa_blok*d_blok**2

I_totaal = I_staaf + I_blok


if __name__ == "__main__":
    print(I_totaal)