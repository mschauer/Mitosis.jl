# Mitosis (work in progress)

Incorporate discrete and continuous time Markov processes as building blocks into probabilistic graphical models.

## Synopsis

Mitosis implements the backward filter and the forward change of measure  of the Automatic Backward Filtering Forward Guiding paradigm  (van der Meulen and Schauer, 2020) as transformation rules for general generative models,  suitable to be incorporated into probabilistic programming approaches.

Starting point is the generative model, a forward description of the probabilistic process dynamics. The backward filter backpropagate the information provided by observations through the model to transform the generative (forward) model into a preconditional model guided by the data. 

The preconditional model approximates the actual conditional model, with known likelihood-ratio between the two and can be used in a variety of sampling and statistical inference approaches to speed up inference.

## References

* Frank van der Meulen, Moritz Schauer (2020): Automatic Backward Filtering Forward Guiding for Markov processes and graphical models. 
