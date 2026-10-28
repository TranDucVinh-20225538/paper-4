# Admin query — ImageNet-1k outside `/data2`

Round-two ID blocked for ImageNet-200 because no ImageNet-1k tree was found
under `/data2` (see `imagenet200_feasibility.md`). Many sites keep ILSVRC under
a shared dataset mount that is not on the same volume as user homes.

**Ask (do not wait on the answer to proceed with CIFAR-100):**

> Does ImageNet-1k (ILSVRC2012 train + val, or an OpenOOD-layout
> `images_largescale/imagenet_1k`) exist anywhere on this system outside
> `/data2` — e.g. `/cm/shared`, a dataset NFS, or another project space we can
> read? If yes, what is the path and the access group?

Record the reply here when it arrives. Until then: CIFAR-100 is round two;
ImageNet-200 remains round three; keep OpenOOD OOD zip downloads in flight.
