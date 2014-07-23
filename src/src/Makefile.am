GRIDTYPE = ALUGRID_CONFORM
GRIDDIM = 2
WORLDDIM = 3
POLORDER = 1
WANT_ISTL = 1
NPARTITION = 2
ADAPTIVE = 0

CLEANFILES = *.log

check_PROGRAMS = partition

AM_CPPFLAGS += $(ALL_PKG_CPPFLAGS) $(DUNEMPICPPFLAGS) -DPOLORDER=$(POLORDER) -DWANT_ISTL=$(WANT_ISTL) -DNPARTITION=$(NPARTITION) -DADAPTIVE=$(ADAPTIVE)
AM_LDFLAGS = $(ALL_PKG_LDFLAGS) $(DUNEMPILDFLAGS)
LDADD = $(ALL_PKG_LIBS) $(DUNEMPILIBS)

partition_SOURCES = partition.cc

EXTRA_DIST =  partition.cc elliptic.hh femscheme.hh heat.hh heatmodel.hh heatscheme.hh model.hh probleminterface.hh rhs.hh temporalprobleminterface.hh
include $(top_srcdir)/am/global-rules
