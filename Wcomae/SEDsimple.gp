set terminal postscript eps size 7.5,5.62 enhanced color \
    font 'Helvetica,20' linewidth 2


set output "SED.eps"

set xrange[1e-6:1e14]
set yrange[1e-9:1e-4]

set format "%g"

unset ytics
unset xtics

set xtics ("eV" 1e0, "keV" 1e3, "MeV" 1e6, "GeV" 1e9, "TeV" 1e12)

set logscale xy
set ylabel "Energy Density"



plot "SSC.dat" with lines lc rgb "#FF0000" smooth sbezier notitle