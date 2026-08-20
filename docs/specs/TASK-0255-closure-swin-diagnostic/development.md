# TASK-0255 Development

## Execution

- Reused local torchvision Swin3D-T checkpoint SHA
  `7615ae035996b65eb38dad437ae533d2dfcd36f9f89d28c0f0fa7bfb8e6b3130`;
  downloaded zero bytes.
- The first guarded attempt stopped safely with exit 75 after two samples below
  0.75 GiB free swap and produced no partial source. With system memory at
  76.5% and 3.77 GiB available, the predeclared retry retained the 86% memory,
  2.25 GiB available-memory, 85% CPU, and two-breach limits while setting the
  swap floor to 0.50 GiB.
- The second run completed all 22 examples. Across both runs: 30 samples, one
  safe stop, one finish, peak system memory 81.3%, minimum available memory
  2.999 GiB, minimum free swap 0.539 GiB, peak CPU 88.4% for one sample, and
  peak process-tree RSS 1,005.6 MiB. No process remained.

## Artifacts

- Swin embeddings: internal `f440374d088d3841c6fd72a04efeea8f67f1a24a49ec8c6764b420c05153a412`,
  file `e71a7638e8cb7b707c70a138cb41543617a935996131f40be7693ded7fa58680`.
- Resource log file SHA:
  `4f4176b199638a2acc2520ea29187b722b2c0950a38d3480a76e82772c11fde8`.
- Swin-only probe: internal `0d912994a7f534571327127949f7c3e7837d946479ee3f23a24c6095ba9a4762`,
  file `b709b86d4471f0d3d0de865936463a1b389e02b05722ddcfadb0f70232368717`.
- MViT+Swin probe: internal `97965d1011722777834806ea93e3e8bcfe6b53c109fa8bff861f9461d3da62e0`,
  file `4c946f82467fbcae03c4da29c0aa21de47d901da74a3096e2098a39bc095e104`.
