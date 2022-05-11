set terminal postscript eps size 7.5,5.62 enhanced color \
    font 'Helvetica,20' linewidth 2


set output "SED.eps"

#set xrange[1e17:1e24]
#set yrange[1e-10:1e-3] 

set format "%g"

set logscale xy
#set ytics 1e-10,1e-9,1e-5
set xlabel "{/Symbol n} [Hz]" font ",30"
set ylabel "{/Symbol n}f_{{/Symbol n}} [erg cm^{-2} s^{-1}]" font ",25"
set key right bottom font ",20"


#	"dataSSC.dat" u 1:2 title "SSC" with lines lw 2 lc 3 lt"..-",\
#	 "Wcomae/wcomae1.dat" u 1:2:5:6 with yerrorbar ps 2 notitle,\
#	 "Wcomae/wcomae2.dat" u 1:2:5:6 with yerrorbar ps 2 notitle,\
#	 "Wcomae/wcomae3.dat" u 1:2:5:6 with yerrorbar ps 2 notitle,\
#	 "Wcomae/wcomae4.dat" u 1:2:5:6 with yerrorbar ps 2 notitle,\
#	 "Wcomae/wcomae5.dat" u 1:2:5:6 with yerrorbar ps 2 notitle,\
#	 "Wcomae/wcomae6.dat" u 1:2:5:6 with yerrorbar ps 2 notitle,\
#	 "Wcomae/wcomae7.dat" u 1:2:5:6 with yerrorbar ps 2 notitle,\
#	 "Wcomae/wcomae8.dat" u 1:2:5:6 with yerrorbar ps 2 notitle,\
#	 "Wcomae/wcomae9.dat" u 1:2:5:6 with yerrorbar ps 2 notitle,\
#	 "Wcomae/wcomae10.dat" u 1:2:5:6 with yerrorbar ps 2 notitle

plot "SSC_Spectra.dat" u 1:($2) title "Synchrotron Self-Compton" with lines lw 2 lc 1 lt"..-",\
	 "Sync_Spectra.dat" u 1:2 title "Synchrotron" with lines lw 2 lc 3 lt"..-",\
	 "WComaeDataSynch_Hz-erg.dat" u 1:2:5:6 with yerrorbar ps 2 notitle,\
	 "WComaeDataSSC_Hz-erg.dat" u 1:2:5:6 with yerrorbar ps 2 notitle
