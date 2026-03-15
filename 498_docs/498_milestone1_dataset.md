
We know that the "Tiny Shakespeare"  dataset had around 1 million characters/tokens with a vocab of 65\. Let's do some quick token math to see what 1 million tokens looks like in the MIDI universe:

* A dense, fast piano piece averages about 10-15 notes per second.  
* Each note translates to roughly 3 tokens in our event vocabulary (NOTE\_ON, NOTE\_OFF, and the intermediate TIME\_SHIFT).  
* That means 1 second of music approx 30 to 45 tokens.  
* 1,000,000 tokens approx 22,000 to 33,000 seconds of music $\\approx$ **6 to 9 hours of continuous piano.**

Looking at the MAESTRO dataset stats, the entire Validation split is 19.4 hours. So, for a \~1MB equivalent dataset, we can just grab about **half of the MAESTRO Validation split** (around 60-70 songs) and concatenate them. This will give the model a beautifully rich dataset filled with diverse arpeggios, scales, and chords, without taking days to train.