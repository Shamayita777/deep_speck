"""
AutoND/SPECK32/64 adapter for Experimental Validity.

A thin integration of the generic EV framework (../framework) with the pinned AutoND
round-5 DBitNet protocol. Data come from the existing AutoNDDatasetAdapter under the
canonical AUTOND_SOURCE_PIN; statistics, decisions, certificates, power simulation,
firewall and resumability are the framework's. The representation intervention is
autond.representation, which depends on no Gohr code.
"""
