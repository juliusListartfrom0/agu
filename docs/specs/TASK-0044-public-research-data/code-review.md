# Code Review

The first implementation incorrectly treated successful YouTube oEmbed metadata
as proof that media were downloadable. A real yt-dlp probe exposed a geographic
restriction. The design now has separate metadata, playback, and local SHA seal
gates. Partial downloads cannot satisfy the local-media lookup, and truth values
remain unavailable to runtime consumers.

Residual risk: upstream videos may disappear or become geographically restricted;
the plan must therefore be regenerated and locally sealed before model freeze.

The downloader never treats `.part`, `.json`, `.ytdl`, temporary merge files,
or separate yt-dlp format streams such as `.f135.mp4` as complete. State is
replaced atomically and bound to the plan hash. The MOT
adapter records only training provenance and geometry summaries; it declares
both runtime use and acceptance-media reuse forbidden. Upstream model code is
not vendored or imported.
