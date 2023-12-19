export PYTHONPATH="$PYTHONPATH:."

PROBLEM=$1
ID=$2
#LM="uni-gg-gbo-without_clique_seeding-conditional,uni-hg-gbo-without_clique_seeding-conditional,mp-hg-gbo-without_clique_seeding-conditional,uni-hg-gbo-with_clique_seeding-conditional,mp-hg-gbo-with_clique_seeding-conditional"
#LM="uni-gg-fb_generic-without_clique_seeding-conditional,uni-hg-fb_generic-without_clique_seeding-conditional,mp-hg-fb_generic-without_clique_seeding-conditional,uni-hg-fb_generic-with_clique_seeding-conditional,mp-hg-fb_generic-with_clique_seeding-conditional"
LM="mp-hg-fb_no_order-without_clique_seeding-conditional,mp-hg-fb-without_clique_seeding-conditional"
#"uni-gg-gbo-without_clique_seeding-non_conditional,uni-fg-gbo-without_clique_seeding-non_conditional,uni-hg-gbo-without_clique_seeding-non_conditional,mp-fg-gbo-without_clique_seeding-non_conditional,mp-hg-gbo-without_clique_seeding-non_conditional,mp-fg-gbo-with_clique_seeding-non_conditional,mp-hg-gbo-with_clique_seeding-non_conditional"
#LM="lt-static-gbo,lt-fb-online-unpruned,lt-fb-online-pruned"
#LM="mcond-hg-gbo,mcond-hg-fb,mcond-hg-fb-generic"
#LM="mcond-hg-fb-generic,mcond-hg-fb"
#OPTIONS=("ucond-gg-gbo,ucond-fg-gbo,ucond-hg-gbo,mcond-hg-gbo" "ucond-gg-fb,ucond-fg-fb,ucond-hg-fb,mcond-hg-fb" "ucond-gg-fb-generic,ucond-fg-fb-generic,ucond-hg-fb-generic,mcond-hg-fb-generic")
#OPTIONS=( "lt-static-gbo,lt-fb-online-unpruned,lt-fb-online-pruned" "ucond-gg-gbo,ucond-fg-gbo,ucond-hg-gbo,mcond-hg-gbo" "ucond-hg-fb,mcond-hg-fb" "ucond-hg-fb-generic,mcond-hg-fb-generic" "full" )

python rvgomea/cmd/run_sweep.py -o data/${PROBLEM}_${ID}_sweep_bbo -p $PROBLEM -l $LM -d 18 -r 30 -s 64
