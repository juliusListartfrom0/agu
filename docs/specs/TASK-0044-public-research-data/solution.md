# Solution

Reuse TrackID3x3 and TeamTrack for player/tracking/ReID pretraining, BARD and
SpaceJam for event/action pretraining, per-file-reviewed Wikimedia portraits for
optional enrollment, and NBA_Games only for post-freeze acceptance metadata.

AGU owns the adapter, immutable hashes, roster coverage calculation, media
availability gates, train/evaluation split, and answer firewall. It does not
redistribute external broadcast media or bind service inference to yt-dlp. The
optional CLI probe distinguishes public metadata visibility from actual media
extraction, while local files must be SHA-sealed before the media gate passes.
