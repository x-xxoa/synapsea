--[[
	https://github.com/mochji/synapsea
	core/layers/backward.lua

	Synapsea, a simple yet powerful machine learning framework for Lua.
	Copyright (C) 2023 mochji

	This program is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.

	This program is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with this program.  If not, see <https://www.gnu.org/licenses/>.

	ewwow backpwopagation
	error = (output - expected) * activation'(output)  | output
	error = (weight_k * error_j) * activation'(output) | hidden
]]--

local activationsModule = require("core.activations")

local backwardModule = {
	dense,
	averagePooling1D,
	averagePooling2D,
	averagePooling3D,
	maxPooling1D,
	maxPooling2D,
	maxPooling3D,
	sumPooling1D,
	sumPooling2D,
	sumPooling3D,
	upSample1D,
	upSample2D,
	upSample3D,
	zeroPad1D,
	zeroPad2D,
	zeroPad3D,
	crop1D,
	crop2D,
	crop3D,
	convolutional1D,
	convolutional2D,
	convolutional3D,
	convolutionalTranspose1D,
	convolutionalTranspose2D,
	convolutionalTranspose3D,
	flatten,
	reshape,
	add1D,
	add2D,
	add3D,
	subtract1D,
	subtract2D,
	subtract3D,
	multiply1D,
	multiply2D,
	multiply3D,
	divide1D,
	divide2D,
	divide3D,
	softmax,
	activate,
	dropOut
}

function backwardModule.dense(args)
	local activation = activationsModule[args.activation]
	local input, weights, bias, alpha, outputSize = args.input, args.weights, args.bias or 0, args.alpha, args.outputSize

	local gradInput   = {}
	local gradWeights = {}
end

return backwardModule