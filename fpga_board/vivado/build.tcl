set root [file normalize [file join [file dirname [info script]] ..]]
set out [file join $root build]
file mkdir $out
set rtl [file join $root rtl]
set generated [file join $root generated]
if {![file exists [file join $generated router_params.svh]]} {
    error "Run tools/board_demo.py export first"
}
read_verilog -sv [glob [file join $rtl *.sv]]
read_xdc [file join $root vivado urbana_router.xdc]
synth_design -top router_top -part xc7s50csga324-1 -include_dirs [list $generated]
report_utilization -file [file join $out utilization_synth.rpt]
opt_design
place_design
route_design
report_utilization -file [file join $out utilization_route.rpt]
report_timing_summary -file [file join $out timing_route.rpt]
report_drc -file [file join $out drc.rpt]
set paths [get_timing_paths -max_paths 1]
if {[llength $paths] == 0} { error "No routed timing path found" }
set wns [get_property SLACK [lindex $paths 0]]
puts "Routed worst path slack: $wns ns"
if {$wns < 0} { error "100 MHz timing is not closed; see timing_route.rpt" }
write_checkpoint -force [file join $out router_top_routed.dcp]
write_bitstream -force [file join $out router_top.bit]
