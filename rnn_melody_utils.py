import urlparse, urllib2, os, midi, random, re, string, sys
import numpy as np

import source

GENRE = 0
COMPOSER = 1
SONG_DATA = 2

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
    def __init__(self, datadir, select_validation_percentage, select_test_percentage, config, works_per_composer=None,
                 single_composer=None, not_read=False):
        self.datadir = datadir
        self.output_ticks_per_quarter_note = 120
        self.single_composer = single_composer
        self.config = config
        self.pointer = {}
        self.pointer['validation'] = 0
        self.pointer['test'] = 0
        self.pointer['train'] = 0
        print('Data loader: datadir: {}'.format(datadir))
        if not not_read:
            self.download_midi_data()
            self.read_data(select_validation_percentage, select_test_percentage, works_per_composer)

    def download_midi_data(self):
        """
        download_midi_data will download a number of midi files, linked from the html
        pages specified in the sources dict, into datadir. There will be one subdir
        per genre, and within each genre-subdir, there will be a subdir per composer.
        Hence, similar to the structure of the sources dict.
        """
        midi_files = {}

        if os.path.exists(os.path.join(self.datadir, 'do-not-redownload.txt')):
            print 'Already completely downloaded, delete do-not-redownload.txt to check for files to download.'
            return
        for genre in source.sources:
            midi_files[genre] = {}
            for composer in source.sources[genre]:
                midi_files[genre][composer] = []
                for url in source.sources[genre][composer]:
                    print url
                    response = urllib2.urlopen(url)
                    # if 'classicalmidi' in url:
                    #  headers = response.info()
                    #  print headers
                    data = response.read()

                    # htmlinks = re.findall('"(  ?[^"]+\.htm)"', data)
                    # for link in htmlinks:
                    #  print 'http://www.classicalmidi.co.uk/'+strip(link)

                    # make urls absolute:
                    urlparsed = urlparse.urlparse(url)
                    data = re.sub('href="\/', 'href="http://' + urlparsed.hostname + '/', data, flags=re.IGNORECASE)
                    data = re.sub('href="(?!http:)', 'href="http://' + urlparsed.hostname + urlparsed.path[
                                                                                            :urlparsed.path.rfind(
                                                                                                '/')] + '/', data,
                                  flags=re.IGNORECASE)
                    # if 'classicalmidi' in url:
                    #  print data

                    links = re.findall('"(http://[^"]+\.mid)"', data)
                    for link in links:
                        cont = False
                        for p in source.ignore_patterns:
                            if p in link:
                                print 'Not downloading links with {}'.format(p)
                                cont = True
                                continue
                        if cont: continue
                        print link
                        filename = link.split('/')[-1]
                        valid_chars = "-_.()%s%s" % (string.ascii_letters, string.digits)
                        filename = ''.join(c for c in filename if c in valid_chars)
                        print genre + '/' + composer + '/' + filename
                        midi_files[genre][composer].append(filename)
                        localdir = os.path.join(os.path.join(self.datadir, genre), composer)
                        localpath = os.path.join(localdir, filename)
                        if os.path.exists(localpath):
                            print 'File exists. Not redownloading: {}'.format(localpath)
                        else:
                            try:
                                response_midi = urllib2.urlopen(link)
                                try:
                                    os.makedirs(localdir)
                                except:
                                    pass
                                data_midi = response_midi.read()
                                if 'DOCTYPE html PUBLIC' in data_midi:
                                    print 'Seems to have been served an html page instead of a midi file. Continuing with next file.'
                                elif 'RIFF' in data_midi[0:9]:
                                    print 'Seems to have been served an RIFF file instead of a midi file. Continuing with next file.'
                                else:
                                    with open(localpath, 'w') as f:
                                        f.write(data_midi)
                            except:
                                print 'Failed to fetch {}'.format(link)
        with open(os.path.join(self.datadir, 'do-not-redownload.txt'), 'w') as f:
            f.write('This directory is considered completely downloaded.')

    def read_data(self, select_validation_percentage, select_test_percentage, works_per_composer):
        """
        read_data takes a datadir with genre subdirs, and composer subsubdirs
        containing midi files, reads them into training data for an rnn-gan model.
        Midi music information will be real-valued frequencies of the
        tones, and intensity taken from the velocity information in
        the midi files.

        returns a list of tuples, [genre, composer, song_data]
        Also saves this list in self.songs.

        Time steps will be fractions of beat notes (32th notes).
        """

        self.genres = sorted(source.sources.keys())
        print('num genres:{}'.format(len(self.genres)))
        if self.single_composer is not None:
            self.composers = [self.single_composer]
        else:
            self.composers = []
            for genre in self.genres:
                self.composers.extend(source.sources[genre].keys())
            if debug == 'overfit':
                self.composers = self.composers[0:1]
            self.composers = list(set(self.composers))
            self.composers.sort()
        print('num composers: {}'.format(len(self.composers)))
        print('limit works per composer: {}'.format(works_per_composer))

        self.songs = {}
        self.songs['validation'] = []
        self.songs['test'] = []
        self.songs['train'] = []

        # max_composers = 2
        # composer_id   = 0
        if select_validation_percentage or select_test_percentage:
            filelist = []
            for genre in self.genres:
                for composer in self.composers:
                    current_path = os.path.join(self.datadir, os.path.join(genre, composer))
                    if not os.path.exists(current_path):
                        print 'Path does not exist: {}'.format(current_path)
                        continue
                    files = os.listdir(current_path)
                    works_read = 0
                    for i, f in enumerate(files):
                        if os.path.isfile(os.path.join(current_path, f)):
                            print('appending {}'.format(os.path.join(os.path.join(genre, composer), f)))
                            filelist.append(os.path.join(os.path.join(genre, composer), f))
                            works_read += 1
                            if works_per_composer is not None and works_read >= works_per_composer:
                                break
            print('filelist len: {}'.format(len(filelist)))
            random.shuffle(filelist)
            print('filelist len: {}'.format(len(filelist)))

            validation_len = 0
            if select_test_percentage:
                validation_len = int(float(select_validation_percentage / 100.0) * len(filelist))
                print('validation len: {}'.format(validation_len))
                source.file_list['validation'] = filelist[0:validation_len]
                print (
                    'Selected validation set (FLAG --select_validation_percentage): {}'
                        .format(source.file_list['validation']))
            if select_test_percentage:
                test_len = int(float(select_test_percentage / 100.0) * len(filelist))
                print('test len: {}'.format(test_len))
                source.file_list['test'] = filelist[validation_len:validation_len + test_len]
                print ('Selected test set (FLAG --select_test_percentage): {}'.format(source.file_list['test']))

        # OVERFIT
        count = 0

        for genre in self.genres:
            # OVERFIT
            if debug == 'overfit' and count > 20: break
            for composer in self.composers:
                # OVERFIT
                if debug == 'overfit' and composer not in self.composers: continue
                if debug == 'overfit' and count > 20: break
                current_path = os.path.join(self.datadir, os.path.join(genre, composer))
                if not os.path.exists(current_path):
                    print 'Path does not exist: {}'.format(current_path)
                    continue
                files = os.listdir(current_path)
                # composer_id += 1
                # if composer_id > max_composers:
                #  print('Only using {} composers.'.format(max_composers))
                #  break
                for i, f in enumerate(files):
                    # OVERFIT
                    if debug == 'overfit' and count > 20: break
                    count += 1

                    if works_per_composer is not None and i >= works_per_composer:
                        break

                    if i % 100 == 99 or i + 1 == len(files) or i + 1 == works_per_composer:
                        print 'Reading files {}/{}: {}'.format(genre, composer, (i + 1))
                    if os.path.isfile(os.path.join(current_path, f)):
                        song_data = self.read_one_file(current_path, f)
                        if song_data is None:
                            continue
                        if os.path.join(os.path.join(genre, composer), f) in source.file_list['validation']:
                            self.songs['validation'].append([genre, composer, song_data])
                        elif os.path.join(os.path.join(genre, composer), f) in source.file_list['test']:
                            self.songs['test'].append([genre, composer, song_data])
                        else:
                            self.songs['train'].append([genre, composer, song_data])
                            # b0reak
        random.shuffle(self.songs['train'])
        self.pointer['validation'] = 0
        self.pointer['test'] = 0
        self.pointer['train'] = 0
        # DEBUG: OVERFIT. overfit.
        if debug == 'overfit':
            self.songs['train'] = self.songs['train'][0:1]
            # print('DEBUG: trying to overfit on the following (repeating for train/validation/test):')
            for i in range(200):
                self.songs['train'].append(self.songs['train'][0])
            self.songs['validation'] = self.songs['train'][0:1]
            self.songs['test'] = self.songs['train'][0:1]
        # print('lens: train: {}, val: {}, test: {}'.format(len(self.songs['train']), len(self.songs['validation']), len(self.songs['test'])))
        return self.songs

    def read_one_file(self, path, filename):
        try:
            if debug:
                print('Reading {}'.format(os.path.join(path, filename)))
            midi_pattern = midi.read_midifile(os.path.join(path, filename))
        except:
            print 'Error reading {}'.format(os.path.join(path, filename))
            return None
        #
        # Interpreting the midi pattern.
        # A pattern has a list of tracks
        # (midi.Track()).
        # Each track is a list of events:
        #   * midi.events.SetTempoEvent: tick, data([int, int, int])
        #     (The three ints are really three bytes representing one integer.)
        #   * midi.events.TimeSignatureEvent: tick, data([int, int, int, int])
        #     (ignored)
        #   * midi.events.KeySignatureEvent: tick, data([int, int])
        #     (ignored)
        #   * midi.events.MarkerEvent: tick, text, data
        #   * midi.events.PortEvent: tick(int), data
        #   * midi.events.TrackNameEvent: tick(int), text(string), data([ints])
        #   * midi.events.ProgramChangeEvent: tick, channel, data
        #   * midi.events.ControlChangeEvent: tick, channel, data
        #   * midi.events.PitchWheelEvent: tick, data(two bytes, 14 bits)
        #
        #   * midi.events.NoteOnEvent:  tick(int), channel(int), data([int,int]))
        #     - data[0] is the note (0-127)
        #     - data[1] is the velocity.
        #     - if velocity is 0, this is equivalent of a midi.NoteOffEvent
        #   * midi.events.NoteOffEvent: tick(int), channel(int), data([int,int]))
        #
        #   * midi.events.EndOfTrackEvent: tick(int), data()
        #
        # Ticks are relative.
        #
        # Tempo are in microseconds/quarter note.
        #
        # This interpretation was done after reading
        # http://electronicmusic.wikia.com/wiki/Velocity
        # http://faydoc.tripod.com/formats/mid.htm
        # http://www.lastrayofhope.co.uk/2009/12/23/midi-delta-time-ticks-to-seconds/2/
        # and looking at some files. It will hopefully be enough
        # for the use in this project.
        #
        # We'll save the data intermediately with a dict representing each tone.
        # The dicts we put into a list. Times are microseconds.
        # Keys: 'freq', 'velocity', 'begin-tick', 'tick-length'
        #
        # 'Output ticks resolution' are fixed at a 32th note,
        #   - so 8 ticks per quarter note.
        #
        # This approach means that we do not currently support
        #   tempo change events.
        #
        # TODO 1: Figure out pitch.
        # TODO 2: Figure out different channels and instruments.
        #

        song_data = []

        # Tempo:
        ticks_per_quarter_note = midi_pattern.resolution
        if ticks_per_quarter_note % self.output_ticks_per_quarter_note != 0:
            return None
        input_ticks_per_output_tick = ticks_per_quarter_note / self.output_ticks_per_quarter_note

        # Multiply with output_ticks_pr_input_tick for output ticks.
        for track in midi_pattern:
            last_event_input_tick = 0
            not_closed_note = None
            for event in track:
                if len(song_data) >= 3000:
                    return song_data
                if type(event) == midi.events.SetTempoEvent:
                    pass  # These are currently ignored
                elif (type(event) == midi.events.NoteOffEvent) or \
                        (type(event) == midi.events.NoteOnEvent and \
                                     event.velocity == 0):
                    if not_closed_note:
                        if event.data[0] == not_closed_note[0]:
                            event_abs_tick = (event.tick + last_event_input_tick) / input_ticks_per_output_tick
                            pitch = not_closed_note[0]
                            if pitch > self.config.melody_params.pitch_max:
                                pitch = pitch - ((pitch - self.config.melody_params.pitch_max) / 12 + 1) * 12
                            elif pitch < self.config.melody_params.pitch_min:
                                pitch = pitch + ((self.config.melody_params.pitch_min - pitch) / 12 + 1) * 12
                            song_data.append(pitch - self.config.melody_params.pitch_min + 2)
                            for i in range((not_closed_note[1] - event_abs_tick) / 15 - 1):
                                song_data.append(1)
                            song_data.append(0)
                            not_closed_note = None
                elif type(event) == midi.events.NoteOnEvent:
                    begin_tick = (event.tick + last_event_input_tick) / input_ticks_per_output_tick
                    note = event.data[0]
                    if not not_closed_note:
                        not_closed_note = [note, begin_tick]
                last_event_input_tick += event.tick
        return song_data

    def rewind(self, part='train'):
        self.pointer[part] = 0

    def get_batch_rnn(self, batchsize, songlength, part='train'):
        """
          get_batch() returns a batch from self.songs, as a
          pair of tensors song_data with shape [batchsize, songlength].

          Since self.songs was shuffled in read_data(), the batch is
          a random selection without repetition.

          A tone  has a feature telling us the pause before it.

        """
        songlength = songlength + 1
        if self.pointer[part] > len(self.songs[part]) - batchsize:
            # return False, [None, None]
            self.pointer[part] = self.pointer[part] % (len(self.songs[part]) - batchsize)
        if self.songs[part]:
            batch = self.songs[part][self.pointer[part]:self.pointer[part] + batchsize]
            self.pointer[part] += batchsize
            batch_songs = np.ndarray(shape=[batchsize, songlength])

            for s in range(len(batch)):
                if len(batch[s][SONG_DATA]) < songlength:
                    raise 'the length of song is too short'
                begin = random.randint(0, len(batch[s][SONG_DATA]) - songlength)
                songmatrix = batch[s][SONG_DATA][begin: begin + songlength]
                batch_songs[s, :] = songmatrix

            return batch_songs[:, 0: songlength - 1], batch_songs[:, 1: songlength]
        else:
            raise 'get_batch() called but self.songs is not initialized.'

    def data_to_song(self, song_name, song_data):
        """
        data_to_song takes a song in internal representation in the shape of
        [song_length] to a midi pattern

        :param song_data: 
        :return: a midi pattern of song_data
        """

        midi_pattern = midi.Pattern([], resolution=int(self.output_ticks_per_quarter_note))
        cur_track = midi.Track([])
        cur_track.append(midi.events.SetTempoEvent(tick=0, bpm=self.config.melody_params.bpm))

        note_not_close = None
        last_note_tick = 0
        for i in range(len(song_data)):
            note = song_data[i]
            if not note_not_close:
                if note > 1:
                    note_not_close = [note, i]
                    event = midi.events.NoteOnEvent(tick=(i - last_note_tick) * 15, velocity=100,
                                                    pitch=note + self.config.melody_params.pitch_min)
                    cur_track.append(event)
            else:
                if note == 0:
                    event = midi.events.NoteOffEvent(tick=(i - note_not_close[1]) * 15, velocity=0,
                                                     pitch=note_not_close[0] + self.config.melody_params.pitch_min)
                    cur_track.append(event)
                    last_note_tick = i
                    note_not_close = None

        if note_not_close:
            event = midi.events.NoteOffEvent(tick=(len(song_data) - note_not_close[1]) * 15, velocity=0,
                                             pitch=note_not_close[0] + self.config.melody_params.pitch_min)
            cur_track.append(event)

        cur_track.append(midi.EndOfTrackEvent(tick=int(self.output_ticks_per_quarter_note)))
        midi_pattern.append(cur_track)
        midi.write_midifile(song_name, midi_pattern)




