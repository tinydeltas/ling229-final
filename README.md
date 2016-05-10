# Final project for LING 229

This is the repo for a final project in Language and Computation II. The authors are Lining Wang, Ian Gonzalez, and David Mcpeek.

## Project Goals

We are creating a classifier that can automatically tag reddit r/relationships posts with a romantic or non-romantic tag.

## Using the code for training and classification

The main classification program is classify_relationship_posts.py.

Main script usage:

`python classify_relationships_posts.py -m [modelfile] -t [training file csv] -e [test file csv] -o [prediction output file]`

Advanced options:

`-p [probability prediction output file] -c [classifer type ("tree" or "logit")]`

Required options: -m, [-e | -t]

## License

This is redistributable under the GNU GPL license (http://www.gnu.org/licenses/gpl-3.0.en.html).

Some of the code was originally written by Jason Baldridge under the GNU GPL (see classify_util.py).