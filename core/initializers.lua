--[[
	https://github.com/x-xxoa/synapsea
	core/initializers.lua

	Synapsea, a simple yet powerful machine learning library made in pure Lua.
	Copyright (C) 2023 x-xxoa
																		   
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

local initializersModule = {
	uniformRandom,
	normalRandom,
	uniformXavier,
	normalXavier,
	uniformHe,
	normalHe,
	constant
}

function initializersModule.uniformRandom(args, index)
	index = index or 1
	local lowerLimit, upperLimit = args.lowerLimit, args.upperLimit
	local shape = args.shape

	local output = {}

	if index == #args.shape then
		for a = 1, args.shape[index] do
			output[a] = mathModule.random.uniform(lowerLimit, upperLimit)
		end
	else
		for a = 1, args.shape[index] do
			output[a] = initializersModule.uniformRandom(
				{
					shape = args.shape,
					lowerLimit = lowerLimit,
					upperLimit = upperLimit
				},
				index + 1
			)
		end
	end

	return output
end

function initializersModule.normalRandom(args, index)
	index = index or 1
	local mean, sd = args.mean, args.sd
	local shape = args.shape

	local output = {}

	if index == #shape then
		for a = 1, shape[index] do
			output[a] = mathModule.random.normal(mean, asd)
		end
	else
		for a = 1, shape[index] do
			output[a] = initializersModule.normalRandom(
				{
					shape = shape,
					mean = mean,
					sd = sd
				},
				index + 1
			)
		end
	end

	return output
end

function initializersModule.uniformXavier(args, index)
	index = index or 1
	local inputs, outputs = args.inputs, args.outputs

	local output = {}

	local limit = math.sqrt(6 / (inputs + outputs))

	if index == #shape then
		for a = 1, shape[index] do
			output[a] = mathModule.random.uniform(-limit, limit)
		end
	else
		for a = 1, shape[index] do
			output[a] = initializersModule.uniformXavier(
				{
					shape = shape,
					inputs = inputs,
					outputs = outputs
				},
				index + 1
			)
		end
	end

	return output
end

function initializersModule.normalXavier(args, index)
	index = index or 1
	local inputs, outputs = args.inputs, args.outputs

	local output = {}

	local sd = math.sqrt(6 / (inputs + outputs))

	if index == #shape then
		for a = 1, shape[index] do
			output[a] = mathModule.random.normal(0, sd)
		end
	else
		for a = 1, shape[index] do
			output[a] = initializersModule.normalXavier(
				{
					shape = shape,
					inputs = inputs,
					outputs = outputs
				},
				index + 1
			)
		end
	end

	return output
end

function initializersModule.uniformHe(args, index)
	index = index or 1
	local inputs = args.inputs

	local output = {}

	local limit = math.sqrt(2 / inputs)

	if index == #shape then
		for a = 1, shape[index] do
			output[a] = mathModule.random.uniform(-limit, limit)
		end
	else
		for a = 1, shape[index] do
			output[a] = initializersModule.uniformHe(
				{
					shape = shape,
					inputs = inputs
				},
				index + 1
			)
		end
	end

	return output
end

function initializersModule.normalHe(args)
	index = index or 1
	local inputs = args.inputs

	local output = {}

	local sd = math.sqrt(2 / inputs)

	if index == #shape then
		for a = 1, shape[index] do
			output[a] = mathModule.random.normal(0, sd)
		end
	else
		for a = 1, shape[index] do
			output[a] = initializersModule.normalHe(
				{
					shape = shape,
					inputs = inputs
				},
				index + 1
			)
		end
	end

	return output
end

function initializersModule.constant(args)
	index = index or 1
	local value = args.value

	local output = {}

	if index == #shape then
		for a = 1, shape[index] do
			output[a] = value
		end
	else
		for a = 1, shape[index] do
			output[a] = initializersModule.constant(
				{
					shape = shape,
					value = value
				},
				index + 1
			)
		end
	end

	return output
end

return initializersModule
