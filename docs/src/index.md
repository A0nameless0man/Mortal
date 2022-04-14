<p align="center">
  <img src="assets/logo.png"/>
</p>

# Mortal

[![GitHub Workflow Status](https://img.shields.io/github/workflow/status/Equim-chan/Mortal/build-libriichi?label=libriichi)](https://github.com/Equim-chan/Mortal/actions)
[![GitHub Workflow Status](https://img.shields.io/github/workflow/status/Equim-chan/Mortal/deploy-docs?label=docs)](https://mortal.ekyu.moe)
[![](https://img.shields.io/github/license/Equim-chan/Mortal)](https://github.com/Equim-chan/Mortal/blob/main/LICENSE)

![GitHub top language](https://img.shields.io/github/languages/top/Equim-chan/Mortal)
![Lines of code](https://img.shields.io/tokei/lines/github/Equim-chan/Mortal)
![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/Equim-chan/Mortal)

Mortal ([凡夫](https://www.mdbg.net/chinese/dictionary?wdqb=%E5%87%A1%E5%A4%AB)) is a free and open source AI for Japanese mahjong, powered by deep reinforcement learning.

The development of Mortal is hosted on GitHub at <https://github.com/Equim-chan/Mortal>.

## Features
* [x] A strong mahjong AI that is compatible with Tenhou's standard ranked rule for four-player mahjong.
* [x] A blazingly fast mahjong emulator written in Rust with a Python interface. Up to 23K hanchans per hour[^env] can be achieved using the Rust emulator combined with Python neural network inference.
* [x] An easy-to-use mjai interface.
* [x] Free and open source.

```admonish note "WIP features"
* [ ] Serve as a backend for mjai-reviewer (formerly known as akochan-reviewer).
* [ ] Limited reasoning support.
```

## About this doc
**This doc is work in progress, so most pages are empty right now.**

## Okay cool now give me the weights!
Read [this post](https://gist.github.com/Equim-chan/cf3f01735d5d98f1e7be02e94b288c56) for details regarding this topic.

## License
### Code
[![AGPL-3.0-or-later](assets/agpl.png)](https://github.com/Equim-chan/Mortal/blob/main/LICENSE)

Copyright (C) 2021-2022 Equim

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.

### Logo and Other Assets
[![CC BY-SA 4.0](assets/by-sa.png)](https://creativecommons.org/licenses/by-sa/4.0/)

[^env]: Measured on NVIDIA® GeForce® RTX 2060 SUPER™ with AMD Ryzen™ 5 3600, game batch size 2000.
