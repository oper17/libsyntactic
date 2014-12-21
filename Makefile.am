lib_LTLIBRARIES = libsyntactic.la
libsyntactic_la_SOURCES = syntactic.h transfer_functions.h syntactic.cc transfer_functions.cc randoom.h
install-am:
	cp .libs/libsyntactic.a $(libdir)/
	mkdir /usr/include/syntactic/
	cp *.h /usr/include/syntactic/
uninstall-am:
	rm -fdr /usr/include/syntactic
	rm $(libdir)/libsyntactic*

test:
	sh maketest.sh
	


