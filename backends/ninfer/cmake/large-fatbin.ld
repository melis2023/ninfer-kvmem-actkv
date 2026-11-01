/* Linker script fragment for builds of several CUDA architectures (CMakeLists.txt). Their device
   code reaches several GiB, and GNU ld places the fatbinary section between the read-only data and
   the writable data by default, which puts .data and .got more than 2 GiB from .text and overflows
   the 32-bit PC-relative references between them. The CUDA runtime finds each fatbinary through a
   64-bit pointer, so the section moves beside the large-data sections at the end of the image. */
SECTIONS
{
  .nv_fatbin : { *(.nv_fatbin) }
}
INSERT AFTER .lrodata;
