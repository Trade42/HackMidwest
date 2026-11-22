set root [file normalize [file join [file dirname [info script]] ..]]
set bit [file join $root build router_top.bit]
if {![file exists $bit]} { error "Build the bitstream first: $bit" }
open_hw_manager
connect_hw_server
open_hw_target
set devices [get_hw_devices *xc7s50*]
if {[llength $devices] != 1} {
    error "Expected one XC7S50 on JTAG, found [llength $devices]: $devices"
}
set dev [lindex $devices 0]
set_property PROGRAM.FILE $bit $dev
program_hw_devices $dev
refresh_hw_device $dev
puts "Programmed $dev with $bit"
