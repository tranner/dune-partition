dnl -*- autoconf -*-
# Macros needed to find dune-partition and dependent libraries.  They are called by
# the macros in ${top_src_dir}/dependencies.m4, which is generated by
# "dunecontrol autogen"

# Additional checks needed to build dune-partition
# This macro should be invoked by every module which depends on dune-partition, as
# well as by dune-partition itself
AC_DEFUN([DUNE_PARTITION_CHECKS])

# Additional checks needed to find dune-partition
# This macro should be invoked by every module which depends on dune-partition, but
# not by dune-partition itself
AC_DEFUN([DUNE_PARTITION_CHECK_MODULE],
[
  DUNE_CHECK_MODULES([dune-partition],[partition/partition.hh])
])
