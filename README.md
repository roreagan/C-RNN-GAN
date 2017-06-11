# Generating Mesuic With RNN.

you can use three kind of models to generate music.


to run:
python melody_generator.py --network gan --datadir data/examples/ --traindir data/traindir2/


--datadir
datadir must be provided where midi files is needed.


--traindir
network must be provided to save network automatically in traindir


--network
you can choose rnn_gan, rnn, seq_gan with this parameters. rnn os default.
