import urlparse, urllib2, os, midi, random, re, string, sys
import numpy as np


# INDICES IN BATCHES (LENGTH,FREQ,VELOCITY are repeated self.tones_per_cell times):
TICKS_FROM_PREV_START = 0
LENGTH = 1
FREQ = 2
VELOCITY = 3

# INDICES IN SONG DATA (NOT YET BATCHED):
BEGIN_TICK = 0

NUM_FEATURES_PER_TONE = 3

debug = ''
# debug = 'overfit'


class MusicDataLoader(object):
    def __init__(self, datadir, config, not_read=False):
        self.datadir = datadir
        self.output_ticks_per_quarter_note = 120
        self.config = config
        self.pointer = {}
        self.pointer['validation'] = 0
        self.pointer['test'] = 0
        self.pointer['train'] = 0
        print('Data loader: datadir: {}'.format(datadir))
        if not not_read:
            self.read_data()


    def read_data(self):
        """
        read_data takes a datadir containing midi files, reads them into training data for an rnn model.
        Midi music information will be a shape aof [2, 0, 3, 0, 5, 0]

        each steps will be fractions of beat notes (32th notes) and each number which is not 0 and 1 is 
        the pitch of the note.
        """

        self.songs = {}
        self.songs['validation'] = []
        self.songs['test'] = []
        self.songs['train'] = []

        files = os.listdir(self.datadir)
        for i, f in enumerate(files):
            song_data = self.read_one_file(self.datadir, f)
            if song_data is None:
                continue
            self.songs['train'].append(song_data)
            print('Read midi %s' % os.path.join(self.datadir, f))

        random.shuffle(self.songs['train'])
        self.pointer['validation'] = 0
        self.pointer['test'] = 0
        self.pointer['train'] = 0
        return self.songs

    def read_one_file(self, path, filename):
        try:
            if debug:
                print('Reading {}'.format(os.path.join(path, filename)))
            midi_pattern = midi.read_midifile(os.path.join(path, filename))
        except:
            print 'Error reading {}'.format(os.path.join(path, filename))
            return None

        song_data = []

        # Tempo:
        ticks_per_quarter_note = midi_pattern.resolution
        if ticks_per_quarter_note % self.output_ticks_per_quarter_note != 0:
            return None
        input_ticks_per_output_tick = ticks_per_quarter_note / self.output_ticks_per_quarter_note

        for track in midi_pattern:
            last_event_input_tick = 0
            not_closed_notes = []
            for event in track:
                if type(event) == midi.events.SetTempoEvent:
                    pass  # These are currently ignored
                elif (type(event) == midi.events.NoteOffEvent) or \
                        (type(event) == midi.events.NoteOnEvent and \
                                     event.velocity == 0):
                    retained_not_closed_notes = []
                    for e in not_closed_notes:
                        if event.data[0] == e[FREQ]:
                            event_abs_tick = (event.tick + last_event_input_tick) / input_ticks_per_output_tick
                            # current_note['length'] = float(ticks*microseconds_per_tick)
                            e[LENGTH] = event_abs_tick - e[BEGIN_TICK]
                            song_data.append(e)
                        else:
                            retained_not_closed_notes.append(e)
                    not_closed_notes = retained_not_closed_notes
                elif type(event) == midi.events.NoteOnEvent:
                    begin_tick = (event.tick + last_event_input_tick) / input_ticks_per_output_tick
                    note = [0] * (NUM_FEATURES_PER_TONE + 1)
                    note[FREQ] = event.data[0]
                    note[VELOCITY] = event.data[1]
                    note[BEGIN_TICK] = begin_tick
                    not_closed_notes.append(note)
                last_event_input_tick += event.tick
        song_data.sort(key=lambda e: e[BEGIN_TICK])
        return song_data

    def rewind(self, part='train'):
        self.pointer[part] = 0

    def get_batch(self, batchsize, songlength, part='train'):
        """
          get_batch() returns a batch from self.songs, as a
          pair of tensors song_data with shape [batchsize, songlength, num_song_features].

          Since self.songs was shuffled in read_data(), the batch is
          a random selection without repetition.

          A tone  has a feature telling us the pause before it.

        """
        if self.pointer[part] > len(self.songs[part]) - batchsize:
            # return False, [None, None]
            self.pointer[part] = self.pointer[part] % (len(self.songs[part]) - batchsize)
        if self.songs[part]:
            batch = self.songs[part][self.pointer[part]:self.pointer[part] + batchsize]
            self.pointer[part] += batchsize
            batch_songs = np.ndarray(shape=[batchsize, songlength, self.config.num_song_features])

            for s in range(len(batch)):
                songmatrix = np.ndarray(shape=[songlength, self.config.num_song_features])

                begin = 1
                if len(batch[s]) > songlength:
                    begin = random.randint(1, len(batch[s]) - songlength - 1)
                else:
                    raise 'data is too short'
                matrixrow = 0
                n = begin
                while matrixrow < songlength:
                    event = np.zeros(shape=[NUM_FEATURES_PER_TONE + 1])
                    length = batch[s][n][LENGTH]
                    if length > self.config.melody_params.length_max:
                        length = self.config.melody_params.length_max
                    elif length < self.config.melody_params.length_min:
                        length = self.config.melody_params.length_min
                    event[LENGTH] = (length - self.config.melody_params.length_min) / 15 + \
                                    int(abs(np.random.normal(0, 1, 1)))

                    pitch = batch[s][n][FREQ]
                    if pitch > self.config.melody_params.pitch_max:
                        pitch = pitch - ((pitch - self.config.melody_params.pitch_max) / 12 + 1) * 12
                    elif pitch < self.config.melody_params.pitch_min:
                        pitch = pitch + ((self.config.melody_params.pitch_min - pitch) / 12 + 1) * 12
                    event[FREQ] = pitch - self.config.melody_params.pitch_min

                    velocity = batch[s][n][VELOCITY]
                    if velocity > self.config.melody_params.velocity_max:
                        velocity = self.config.melody_params.velocity_max
                    elif velocity < self.config.melody_params.velocity_min:
                        velocity = self.config.melody_params.velocity_min
                    event[VELOCITY] = velocity - self.config.melody_params.velocity_min

                    ticks = batch[s][n][TICKS_FROM_PREV_START] - batch[s][n-1][TICKS_FROM_PREV_START]
                    if ticks > self.config.melody_params.ticks_max:
                        ticks = self.config.melody_params.ticks_max
                    elif ticks < self.config.melody_params.ticks_min:
                        ticks = self.config.melody_params.ticks_min
                    event[TICKS_FROM_PREV_START] = (ticks - self.config.melody_params.ticks_min) / 15 + \
                                                   int(abs(np.random.normal(0, 1, 1)))

                    songmatrix[matrixrow, :] = event
                    matrixrow += 1
                    n += 1
                batch_songs[s, :, :] = songmatrix

            return batch_songs
        else:
            raise 'get_batch() called but self.songs is not initialized.'


    def data_to_song(self, song_name, song_data):
        """
        data_to_song takes a song in internal representation in the shape of
        [song_length, num_song_features] to a midi pattern
        all the features are retrieved to the settings in config 
        
        :param song_data: 
        :return: a midi pattern of song_data
        """

        midi_pattern = midi.Pattern([], resolution=int(self.output_ticks_per_quarter_note))
        cur_track = midi.Track([])
        cur_track.append(midi.events.SetTempoEvent(tick=0, bpm=self.config.melody_params.bpm))

        song_events_absolute_ticks = []
        abs_tick_note_beginning = 0
        for frame in song_data:
            ticks = int(round(frame[TICKS_FROM_PREV_START])) * 15 + self.config.melody_params.ticks_min
            if ticks > self.config.melody_params.ticks_max:
                ticks = self.config.melody_params.ticks_max
            abs_tick_note_beginning += ticks
            tick_len = int(round(frame[LENGTH])) * 15 + self.config.melody_params.length_min
            if tick_len > self.config.melody_params.length_max:
                tick_len = self.config.melody_params.length_max
            pitch = int(round(frame[FREQ])) + self.config.melody_params.pitch_min
            if pitch > self.config.melody_params.pitch_max:
                pitch = self.config.melody_params.pitch_max
            velocity = int(round(frame[VELOCITY])) + self.config.melody_params.velocity_min
            if velocity > self.config.melody_params.velocity_max:
                velocity = self.config.melody_params.velocity_max

            song_events_absolute_ticks.append((abs_tick_note_beginning,
                                               midi.events.NoteOnEvent(
                                                   tick=0,
                                                   velocity=velocity,
                                                   pitch=pitch)))
            song_events_absolute_ticks.append((abs_tick_note_beginning + tick_len,
                                               midi.events.NoteOffEvent(
                                                   tick=0,
                                                   velocity=0,
                                                   pitch=pitch)))

        song_events_absolute_ticks.sort(key=lambda e: e[0])
        abs_tick_note_beginning = 0
        for abs_tick, event in song_events_absolute_ticks:
            rel_tick = abs_tick - abs_tick_note_beginning
            event.tick = rel_tick
            cur_track.append(event)
            abs_tick_note_beginning = abs_tick

        cur_track.append(midi.EndOfTrackEvent(tick=int(self.output_ticks_per_quarter_note)))
        midi_pattern.append(cur_track)
        midi.write_midifile(song_name, midi_pattern)




