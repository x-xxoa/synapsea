--[[
	https://github.com/mochji/synapsea
	core/model.lua

	Synapsea, a simple yet powerful machine learning library made in pure Lua.
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
]]--

local buildModule = require(SYNAPSEA_PATH .. "core.layers.build")
local layersModule = require(SYNAPSEA_PATH .. "core.layers.layers")
local initializersModule = require(SYNAPSEA_PATH .. "core.initializers")

local modelModule = {
	layerToParameters,
	addLayer,
	removeLayer,
	summary,
	initialize,
	forwardPass,
	fit,
	new,
	export,
	import
}

function modelModule.layerToParameters(layer)
	local parameters = {}

	if layer.parameters then
		for parameter, _ in pairs(layer.parameters) do
			parameters[parameter] = layer.parameters[config]
		end
	end

	return parameters
end

function modelModule.addLayer(model, layerType, buildParameters, layerNumber)
	buildParameters, layerNumber = buildParameters or {}, layerNumber or #model.layerConfig + 1

	local layer

	if buildModule[layerType] then
		-- Get last layer output shape

		if layerNumber == 1 then
			buildParameters.inputShape = model.inputShape
		else
			buildParameters.inputShape = model.layerConfig[layerNumber - 1].outputShape
		end

		layer, parameterBuild = buildModule[layerType](buildParameters)
		table.insert(model.parameterBuild, layerNumber, parameterBuild or {})
	else
		layer = buildParameters
	end

	layer.type = layerType
	table.insert(model.layerConfig, layerNumber, layer)

	return model
end

function modelModule.removeLayer(model, layerNumber)
	layerNumber = layerNumber or #model.layerConfig

	assert(model.layerConfig[layerNumber], string.format("Attempt to remove a non-existant layer (%d).", layerNumber))

	table.remove(model.layerConfig[layerNumber])
	table.remove(model.parameterBuild[layerNumber])

	return model
end

function modelModule.summary(model, toString)
end

function modelModule.initialize(model, optimizer, optimizerParameters, regularizer, regularizerParameters)
	-- Create parameters in layers and initialize

	if model.parameterBuild then
		for a = 1, #model.parameterBuild do
			local layer = model.layerConfig[a]

			for parameterName, parameter in pairs(model.parameterBuild[a]) do
				layer.parameters[parameterName] = initializersModule[layer.initializer[parameterName].initializer](
					model.parameterBuild[a][parameterName].shape,
					layer.initializer[parameterName].parameters
				)
			end
		end
	end

	-- Create training parameters

	if optimizer then
		model.trainingConfig.optimizer = {
			optimizer = optimizer,
			parameters = optimizerParameters
		}
	end

	if regularizer then
		model.trainingConfig.regularizer = {
			regularizer = regularizer,
			parameters = regularizerParameters
		}
	end

	if not model.layerConfig[#model.layerConfig] then
		model.outputShape = model.inputShape
	else
		model.outputShape = model.layerConfig[#model.layerConfig].outputShape
	end

	model.addLayer, model.parameterBuild = nil, nil
	model.layers = #model.layerConfig
	model.forwardPass = modelModule.forwardPass

	return model
end

function modelModule.forwardPass(model, input)
	local lastOutput = input

	for a = 1, #model.layerConfig do
		local parameters = modelModule.layerToParameters(model.layerConfig[a])
		parameters.input = lastOutput

		lastOutput = layersModule[model.layerConfig[a].type](parameters)
	end

	return lastOutput
end

function modelModule.fit(model, dataset, epochs, callBacks)
end

function modelModule.new(inputShape, metaData)
	local model = {
		metaData = metaData or {},
		inputShape = inputShape,
		parameterBuild = {},
		layerConfig = {},
		trainingConfig = {}
	}

	model.metaData.synapseaVersion = SYNAPSEA_VERSION

	-- Model functions

	model.addLayer = modelModule.addLayer
	model.removeLayer = modelModule.removeLayer
	model.export = modelModule.export
	model.summary = modelModule.summary
	model.initialize = modelModule.initialize

	return model
end

function modelModule.export(model, fileName)
	local f, err = io.open(fileName, "w")

	assert(f, fileName .. ": " .. err)

	f:write("return 'im so sorry but say goodbye to your model, this isn't done yet :('")
	f:close()
end

function modelModule.import(fileName)
	local model = dofile(fileName)

	model.removeLayer = modelModule.removeLayer
	model.initialize = modelModule.initialize
	model.export = modelModule.export

	if model.parameterBuild then
		model.addLayer = modelModule.addLayer
	else
		model.forwardPass = modelModule.forwardPass
		model.fit = modelModule.fit
	end

	return model
end

return modelModule
