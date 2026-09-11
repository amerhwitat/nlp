# Dynamic Partition Analysis

Dynamic partitions are represented by metadata in `super` images. FlashTool models them separately from physical GPT partitions so logical partition size and group membership can be evaluated before a write.

## Planned analyzer capabilities

- Identify `super.img` / `super_empty.img` artifacts.
- Parse bounded liblp metadata.
- List logical partitions and groups.
- Calculate used/free space within groups.
- Detect read-only flags and metadata slots.
- Compare logical image sizes with available group capacity.

## Safety

Metadata inspection is read-only. FlashTool must never assume that a logical partition exists because an image filename contains the partition name.

## Source

Android Open Source Project dynamic partition tools and `system/core` liblp are the reference sources.
