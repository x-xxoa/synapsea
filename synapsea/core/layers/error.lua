--[[
	https://github.com/mochji/synapsea
	core/layers/error.lua

	Synapsea, a simple yet powerful machine learning framework for Lua.
	Copyright (C) 2024 mochji

	MIT License
]]--

local activationsModule = require("core.activations")

local errorModule = {
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

function errorModule.dense(args)
	local layerOutput, weights, alpha, outputSize, forwardError = args.output, args.weights, args.alpha, args.outputSIze, args.forwardError
	local activation = activationsModule[args.activation]

	local inputSize = #weights

	local output = {}

	for a = 1, outputSize do
		output[a] = {}

		for b = 1, inputSize do
			output[a][b] = output[a][b] + weights[b][a] * forwardError[a] * activation(layerOutput[a], true, alpha)
		end
	end

	return output
end

return errorModule
