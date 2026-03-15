# The MAESTRO Dataset

Oct 29, 2018

**MAESTRO** (MIDI and Audio Edited for Synchronous TRacks and Organization) is a dataset composed of about 200 hours of virtuosic piano performances captured with fine alignment (\~3 ms) between note labels and audio waveforms.

# Dataset

We partnered with organizers of the [International Piano-e-Competition](http://piano-e-competition.com/) for the raw data used in this dataset. During each installment of the competition virtuoso pianists perform on Yamaha Disklaviers which, in addition to being concert-quality acoustic grand pianos, utilize an integrated high-precision MIDI capture and playback system. Recorded MIDI data is of sufficient fidelity to allow the audition stage of the competition to be judged remotely by listening to contestant performances reproduced over the wire on another Disklavier instrument.

The dataset contains about 200 hours of paired audio and MIDI recordings from ten years of International Piano-e-Competition. The MIDI data includes key strike velocities and sustain/sostenuto/una corda pedal positions. Audio and MIDI files are aligned with ∼3 ms accuracy and sliced to individual musical pieces, which are annotated with composer, title, and year of performance. Uncompressed audio is of CD quality or higher (44.1–48 kHz 16-bit PCM stereo).

A train/validation/test split configuration is also proposed, so that the same composition, even if performed by multiple contestants, does not appear in multiple subsets. Repertoire is mostly classical, including composers from the 17th to early 20th century.

For more information about how the dataset was created and several applications of it, please see the paper where it was introduced: [Enabling Factorized Piano Music Modeling and Generation with the MAESTRO Dataset](https://goo.gl/magenta/maestro-paper).

For an example application of the dataset, see our blog post on [Wave2Midi2Wave](https://g.co/magenta/wave2midi2wave).

# Download

MAESTRO is provided as a zip file containing the MIDI and WAV files as well as metadata in CSV and JSON formats. A MIDI-only archive of the dataset is also available.

The metadata files have the following fields for every MIDI/WAV pair:

| Field | Description |
| ----- | ----- |
| canonical\_composer | Composer of the piece. We have attempted to standardize on a single spelling for a given name. |
| canonical\_title | Title of the piece. Not guaranteed to be standardized to a single representation. |
| split | Suggested train/validation/test split. |
| year | Year of performance. |
| midi\_filename | MIDI filename. |
| audio\_filename | WAV filename. |
| duration | Duration in seconds, based on the MIDI file. |

### [maestro-v3.0.0.zip](https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.zip)

## **Size: 101GB (120GB uncompressed)** **SHA256: `6680fea5be2339ea15091a249fbd70e49551246ddbd5ca50f1b2352c08c95291`**

### [maestro-v3.0.0-midi.zip](https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip)

## **Size: 56MB (81MB uncompressed)** **SHA256: `70470ee253295c8d2c71e6d9d4a815189e35c89624b76d22fce5a019d5dde12c`**

## **Metadata files as separate downloads:**

* ## [**maestro-v3.0.0.csv**](https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.csv)

* ## [**maestro-v3.0.0.json**](https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.json)

## **Certain statistics of the dataset:**

| Split | Performances | Duration (hours) | Size (GB) | Notes (millions) |
| ----- | ----: | ----: | ----: | ----: |
| **Train** | **962** | **159.2** | **96.3** | **5.66** |
| **Validation** | **137** | **19.4** | **11.8** | **0.64** |
| **Test** | **177** | **20.0** | **12.1** | **0.74** |
| **Total** | **1276** | **198.7** | **120.2** | **7.04**  |

## 