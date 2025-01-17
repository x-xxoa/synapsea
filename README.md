<p align="center">
    <img src="https://github.com/mochji/synapsea/assets/117334318/1c07a93b-5e83-4a1e-891e-632e7b460f87">
</p>

### **Note: This is a development release!**

[![License: GPL v3](https://img.shields.io/badge/license-MIT-blue)](https://www.opensource.org/license/mit)
[![Lua](https://img.shields.io/badge/Lua-5.4%2B-blueviolet)](https://www.lua.org/)
[![Release](https://img.shields.io/github/v/release/mochji/synapsea)](https://github.com/mochji/synapsea/releases)

## Overview


Synapsea is a simple yet powerful machine learning framework designed for building, saving, training and running powerful machine learning models.

Synapsea is built from the ground up to be simple, easy to understand and portable with no external libraries, requring only a Lua interpreter.

Synapsea runs entirely on the CPU and is single-threaded, this is a limitation but one that makes Synapsea easy to embed wherever.

## Table of Contents

 - [Installation](#installation)
 - [Usage](#usage)
 - [Examples](#examples)
 - [Documentation](#documentation)
 - [Contributing](#contributing)
 - [Version Numbering](#version-numbering)
 - [License](#license)
 - [Also](#also)

## Installatation

Synapsea is very simple to install, either clone the GitHub repository using Git:

```
git clone https://github.com/mochji/synapsea.git
```

Or download the zip file from GitHub and unzip it.

Once you have cloned the Synapsea repository just move or copy `synapsea/synapsea` to a directory where you can `require` it.

## Usage

To use Synapsea, just `require` the library:

```lua
local synapsea = require("synapsea")
```

## Examples

Try the Synapsea API:

```lua
> synapsea = require("synapsea")
> synapsea.version
v2.0.0-development
> synapsea.path
/usr/share/lua/5.4/synapsea/
> synapsea.activations.sigmoid(tonumber(io.read()))
2.9
0.94784643692158
> input = synapsea.initializers.normalRandom({3, 3}, {mean = 0, sd = 0.1})
> model = synapsea.Sequential({3, 3}, {"what is this?" = "metadata!"})
> model:addLayer("flatten")
> model:build()
> output = model:forwardPass(input)
> for a = 1, #output do print(a, output[a]) end
1       0.12884759243026
2       0.16374163630011
3       0.076140250714198
4       0.078038014880484
5       0.037395010665463
6       -0.13098177989544
7       0.032318751265418
8       -0.14966043873079
9       0.33041979998093
```

## Version Numbering

Synapsea follows the semantic versioning convention for version numbering.

- **Major Version (X):** Increments for changes that are or may be backward-incompatible or major feature additions.
- **Minor Version (Y):** Increments for features and enhancements that are backward compatible.
- **Patch Version (Z):** Increments for bug fixes, minor changes or minor improvements.
- **Branch Name:** If the branch is not `stable`, then append `-${branchName}` where `branchName` is the name of the branch.

For example:

- **v2.0.0-development**
- **v2.1.11**
- **v3.5.0-rc.1**

## License

This library is licensed under the [MIT License](https://www.opensource.org/license/mit).

## Also

ok, 2 more things!

tldr: ai is stupid, synapsea started as a hyperfixation

 - i gotta be honest synapsea started out as a hyperfixation and now i cant stop working on it because im in too deep
 - i hate ai, i think machine learning for data analysis and processing is cool but i hate quote unquote ""ai"". its stupid, most of the time its just half-baked gpt 6b models and its so stupid and pointless. i don't want synapsea to be seen as one of those or just to hop on the bandwagon. i actually started working on synapsea before ""ai"" blew up.
