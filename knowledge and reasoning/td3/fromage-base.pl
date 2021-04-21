
:- op(1200, xfy, ::).
:- op(1000, fx, si).
:- op(990, xfx, alors).
:- op(980, xfy, et).
:- op(600, yfy, est).
:- op(600, yfy, user:('n est pas')).

rule1 :: si Fromage est mou
et Fromage est fleuri
et Fromage est vache
et diametre(Fromage, 10)
alors Fromage est camembert.

rule2 :: si Fromage est mou
et Fromage est jaune
et Fromage 'n est pas' cuit
et diametre(Fromage, DT)
et DT > 5
alors Fromage est st_nectaire.

rule3 :: si Fromage est gras
alors mg(Fromage, MG)
et MG > 60.

rule4 :: si Fromage est chevre
alors mg(Fromage, 45).

rule5 :: si Fromage 'n est pas' a_trous
alors dt(Fromage, 0).

rule6 :: si Fromage 'n est pas' chevre
et Fromage 'n est pas' vache
alors Fromage est brebis.

rule7 :: si Fromage est dur
et Fromage est brebis
et dt(Fromage, DT)
et DT < 5
et Fromage est jaune
alors Fromage est pyrenees.

rule8 :: si dt(Fromage, DT)
et DT > 4
et DT < 10
et Fromage est cuit
alors Fromage est gruyère.

rule9 :: si Fromage est bleu
et Fromage est brebis
alors Fromage est roquefort.

rule10 :: si Fromage est vache
et dt(Fromage, DT)
et DT < 5
et Fromage est cuit
alors Fromage est comte.

rule11 :: si Fromage 'n est pas' brebis
et Fromage 'n est pas' chevre
alors Fromage est vache.

rule12 :: si Fromage est lave
et dm(Fromage, DM)
et DM > 10
alors Fromage 'n est pas' vache.

rule13 :: si Fromage est jaune
alors Fromage 'n est pas' bleu.

rule14 :: si dm(Fromage, DM)
et DM > 20
et Fromage est bleu
et Fromage est lave
alors Fromage est gorgonzola.

rule15 :: si Fromage est mou
et Fromage est vache
et Fromage est lave
et mt(Fromage, 40)
alors Fromage est livarot.

rule16 :: si Fromage est mou
et Fromage est lave
et Fromage est vache
et dm(Fromage, DM)
et DM > 15
alors Fromage est maroilles.

rule17 :: si Fromage est mou
et Fromage est fleuri
et Fromage est vache
et dm(Fromage, DM)
et DM > 20
alors Fromage est brie.

rule18 :: si Fromage est maigre
alors mt(Fromage, MT)
et MT < 20.

rule19 :: si Fromage est dur
et Fromage est jaune
alors Fromage est cuit.

rule20 :: si Fromage est bleu
et Fromage 'n est pas' fourme
alors Fromage est lave.

rule21 :: si Fromage est cuit
et Fromage 'n est pas' a_trous
alors Fromage est hollande.

rule22 :: si Fromage est hollande
et Fromage 'n est pas' gouda
alors Fromage est edam.

rule23 :: si Fromage est cuit
alors Fromage est dur.

rule24 :: si dm(Fromage, DM)
et DM < 10
et Fromage est dur
et Fromage 'n est pas' cuit
alors Fromage est chevre.

rule25 :: si Fromage est chevre
alors Fromage 'n est pas' a_trous.

rule26 :: si Fromage est bleu
et Fromage est mou
et Fromage est lave
alors Fromage est bleu_d_auvergne.

rule27 :: si dm(Fromage, DM)
et DM < 5
et Fromage est chevre
alors Fromage est picodon.
