"""
AutoND/SPECK32/64 adapter for Experimental Validity.

A thin integration of the generic EV framework (../framework) with the pinned AutoND
round-5 DBitNet protocol. Data come from the existing AutoNDDatasetAdapter under the
canonical AUTOND_SOURCE_PIN; statistics, decisions, certificates, power simulation,
firewall and resumability are the framework's. The Candidate-1 representation transform
is reused from gohr.representation: AutoND's input is the same four 16-bit ciphertext
words, MSB-first, in the same order.
"""
