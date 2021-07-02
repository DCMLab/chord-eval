#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from metric import get_distance
from data_types import ChordType



#%%

distance = 'SPS'
d = get_distance(distance=distance, root1=0, root2=0,
                 chord_type1=ChordType.MAJOR, chord_type2=ChordType.MAJ_MAJ7)
print('{} distance between Cmaj and Cmaj_maj7 : {:.5}'.format(distance, d))

d = get_distance(distance=distance, root1=0, root2=0,
                 chord_type1=ChordType.MAJOR, chord_type2=ChordType.MAJ_MAJ7)
print('{} distance between Cmaj and Cmaj_maj7 : {:.5}'.format(distance, d))

##

d = get_distance(distance=distance, root1=0, root2=1,
                 chord_type1=ChordType.MAJOR, chord_type2=ChordType.MAJOR)
print('{} distance between Cmaj and Dbmaj : {:.5}'.format(distance, d))

#caching
d = get_distance(distance=distance, root1=0, root2=1,
                 chord_type1=ChordType.MAJOR, chord_type2=ChordType.MAJOR)
print('{} distance between Cmaj and Dbmaj : {:.5}'.format(distance, d))

##

d = get_distance(distance=distance, root1=0, root2=1,
                 chord_type1=ChordType.MAJOR, chord_type2=ChordType.MINOR)
print('{} distance between Cmaj and Dbmin : {:.5}'.format(distance, d))

#caching
d = get_distance(distance=distance, root1=0, root2=1,
                 chord_type1=ChordType.MAJOR, chord_type2=ChordType.MINOR)
print('{} distance between Cmaj and Dbmin : {:.5}'.format(distance, d))

##

d = get_distance(distance=distance, root1=0, root2=1,
                 chord_type1=ChordType.MAJOR, chord_type2=ChordType.MINOR,
				 inversion2=1)
print('{} distance between Cmaj and Dbmin : {:.5}'.format(distance, d))

#caching
d = get_distance(distance=distance, root1=0, root2=1,
                 chord_type1=ChordType.MAJOR, chord_type2=ChordType.MINOR,
                 inversion2=1)
print('{} distance between Cmaj and Dbmin : {:.5}'.format(distance, d))

##

d = get_distance(distance=distance, root1=0, root2=1,
                 chord_type1=ChordType.MAJOR, chord_type2=ChordType.MINOR,
                 inversion2=1, transform='cqt')
print('{} distance between Cmaj and Dbmin : {:.5}'.format(distance, d))

#caching
d = get_distance(distance=distance, root1=0, root2=1,
                 chord_type1=ChordType.MAJOR, chord_type2=ChordType.MINOR,
                 inversion2=1, transform='cqt')
print('{} distance between Cmaj and Dbmin : {:.5}'.format(distance, d))

##

d = get_distance(distance=distance, root1=0, root2=1,
                 chord_type1=ChordType.MAJOR, chord_type2=ChordType.MINOR,
                 transform='cqt')
print('{} distance between Cmaj and Dbmaj : {:.5}'.format(distance, d))

#caching
d = get_distance(distance=distance, root1=0, root2=1,
                 chord_type1=ChordType.MAJOR, chord_type2=ChordType.MINOR,
                 transform='cqt')
print('{} distance between Cmaj and Dbmaj : {:.5}'.format(distance, d))


#%%

print('_____________________')

distance = 'voice leading'
d = get_distance(distance=distance, root1=0, root2=0,
                 chord_type1=ChordType.MAJOR, chord_type2=ChordType.MAJ_MAJ7)
print('{} distance between Cmaj and Cmaj_maj7 : {:.5}'.format(distance, d))

d = get_distance(distance=distance, root1=0, root2=0,
                 chord_type1=ChordType.MAJOR, chord_type2=ChordType.MAJ_MIN7,
                 inversion1=0, inversion2=2)
print('{} distance between Cmaj and Cmaj_min7 : {:.5}'.format(distance, d))

d = get_distance(distance=distance, root1=0, root2=6,
                 chord_type1=ChordType.MAJOR, chord_type2=ChordType.MAJOR)
print('{} distance between Cmaj and F#maj : {:.5}'.format(distance, d))

d = get_distance(distance=distance, root1=0, root2=11,
                 chord_type1=ChordType.MAJ_MAJ7, chord_type2=ChordType.MAJ_MAJ7,
                 inversion1=3)
print('{} distance between Cmaj_maj7 and Bmaj_maj7inv1 : {:.5}'.format(distance, d))


d = get_distance(distance=distance, root1=0, root2=11,
                 chord_type1=ChordType.MAJ_MAJ7, chord_type2=ChordType.MAJ_MAJ7)
print('{} distance between Cmaj_maj7 and Bmaj_maj7 : {:.5}'.format(distance, d))


d = get_distance(distance=distance, root1=0, root2=7,
                 chord_type1=ChordType.MAJOR, chord_type2=ChordType.DIMINISHED,
                 inversion2=2, bass_weight=3)
print('{} distance between Cmaj and Gdim : {:.5}'.format(distance, d))

print('_____________________')

distance = 'voice leading'
d = get_distance(distance=distance, root1=0, root2=0,
                 chord_type1=ChordType.MAJOR, chord_type2=ChordType.MAJ_MAJ7)
print('{} distance between Cmaj and Cmaj_maj7 : {:.5}'.format(distance, d))

d = get_distance(distance=distance, root1=0, root2=0,
                 chord_type1=ChordType.MAJOR, chord_type2=ChordType.MAJ_MIN7,
                 inversion1=0, inversion2=2)
print('{} distance between Cmaj and Cmaj_min7 : {:.5}'.format(distance, d))

d = get_distance(distance=distance, root1=0, root2=6,
                 chord_type1=ChordType.MAJOR, chord_type2=ChordType.MAJOR)
print('{} distance between Cmaj and F#maj : {:.5}'.format(distance, d))

d = get_distance(distance=distance, root1=0, root2=11,
                 chord_type1=ChordType.MAJ_MAJ7, chord_type2=ChordType.MAJ_MAJ7,
                 inversion1=3)
print('{} distance between Cmaj_maj7 and Bmaj_maj7inv1 : {:.5}'.format(distance, d))


d = get_distance(distance=distance, root1=0, root2=11,
                 chord_type1=ChordType.MAJ_MAJ7, chord_type2=ChordType.MAJ_MAJ7)
print('{} distance between Cmaj_maj7 and Bmaj_maj7 : {:.5}'.format(distance, d))


d = get_distance(distance=distance, root1=0, root2=7,
                 chord_type1=ChordType.MAJOR, chord_type2=ChordType.DIMINISHED,
                 inversion2=2, bass_weight=3)
print('{} distance between Cmaj and Gdim : {:.5}'.format(distance, d))


#%%

print('_____________________')

distance = 'tone by tone'
d = get_distance(distance=distance, root1=0, root2=0,
                 chord_type1=ChordType.MAJOR, chord_type2=ChordType.MAJ_MAJ7)
print('{} distance between Cmaj and Cmaj_maj7 : {:.5}'.format(distance, d))


d = get_distance(distance=distance, root1=0, root2=0,
                 chord_type1=ChordType.MAJOR, chord_type2=ChordType.MAJ_MAJ7,
                 root_weight=2, bass_weight=3)
print('{} distance between Cmaj and Cmaj_maj7 (weighted): {:.5}'.format(distance, d))

d = get_distance(distance=distance, root1=0, root2=1,
                 chord_type1=ChordType.MAJOR, chord_type2=ChordType.MAJOR)
print('{} distance between Cmaj and Dbmaj : {:.5}'.format(distance, d))

d = get_distance(distance=distance, root1=0, root2=4,
                 chord_type1=ChordType.MAJ_MAJ7, chord_type2=ChordType.MINOR,
                 inversion1=3, inversion2=2)
print('{} distance between Cmaj_maj7in3 and Emin7inv2 : {:.5}'.format(distance, d))

d = get_distance(distance=distance, root1=0, root2=4,
                 chord_type1=ChordType.MAJ_MAJ7, chord_type2=ChordType.MINOR,
                 inversion1=3, inversion2=2, root_weight=2, bass_weight=3)
print('{} distance between Cmaj_maj7in3 and Emin7inv2 (weighted): {:.5}'.format(distance, d))

d = get_distance(distance=distance, root1=0, root2=9,
                 chord_type1=ChordType.MAJOR, chord_type2=ChordType.MIN_MIN7)
print('{} distance between Cmaj and Amin7 : {:.5}'.format(distance, d))

#%%

print('_____________________')

distance = 'SPS'
d = get_distance(distance=distance, root1=0, root2=9,
                 chord_type1=ChordType.MAJ_MIN7, chord_type2=ChordType.MAJOR,
                 equivalences = "octatonic", transform='vqt')
print('{} distance between Cmaj_min7 and Amaj : {:.5}'.format(distance, d))

d = get_distance(distance=distance, root1=0, root2=9,
                 chord_type1=ChordType.MAJ_MIN7, chord_type2=ChordType.MAJ_MIN7,
                 equivalences = "octatonic", transform='vqt')
print('{} distance between Cmaj_min7 and Amaj_min7 : {:.5}'.format(distance, d))

distance = 'tone by tone'
d = get_distance(distance=distance, root1=0, root2=4,
                 chord_type1=ChordType.MAJ_MIN7, chord_type2=ChordType.MAJOR,
                 equivalences = "hexatonic")
print('{} distance between Cmaj_min7 and Emaj : {:.5}'.format(distance, d))

d = get_distance(distance='binary', root1=0, root2=4,
                 chord_type1=ChordType.MAJ_MIN7, chord_type2=ChordType.MAJ_MIN7)
print('{} distance between Cmaj_min7 and Emaj_min7 : {:}'.format(distance, d))
