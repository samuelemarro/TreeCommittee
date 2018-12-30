# TreeCommittee

TreeCommittee is an optimized inference tool that only runs the necessary layers of a neural network.
This is achieved by splitting the network in "bmodules".

## Modules

A smodule is a collection of layers (the "hidden submodule") with the addition of an output layer (the "output submodule").
When performing inference with the network, we apply the first hidden submodule and store its result. We then apply the corresponding output submodule and compare the highest confidence with the module's threshold. If the highest confidence is higher than the threshold, we terminate the execution and use the bmodule's output as the final output; if it is lower, we apply the following hidden submodule to the result of the _hidden_ submodule. This process is repeated until either a module reaches a sufficient confidence or the last module is executed. In the latter case, we use the last output submodule as the final output.
