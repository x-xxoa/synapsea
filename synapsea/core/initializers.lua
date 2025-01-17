--[[
	https://github.com/mochji/synapsea
	core/initializers.lua

	Synapsea, a simple yet powerful machine learning framework for Lua.
	Copyright (C) 2024 mochji

	MIT License
]]--

local initializersModule = {
	zeros,
	ones,
	constant,
	uniformRandom,
	normalRandom,
	uniformXavier,
	normalXavier,
	uniformHe,
	normalHe
}

function initializersModule.zeros(shape)
	local function initializerFunc(shape, index)
		local output = {}

		if index == #shape then
			for a = 1, shape[index] do
				output[a] = 0
			end
		else
			for a = 1, shape[index] do
				output[a] = initializerFunc(
					shape,
					index + 1
				)
			end
		end

		return output
	end

	return initializerFunc(shape, 1)
end

function initializersModule.ones(shape, args)
	local function initializerFunc(shape, index)
		local output = {}

		if index == #shape then
			for a = 1, shape[index] do
				output[a] = 1
			end
		else
			for a = 1, shape[index] do
				output[a] = initializerFunc(
					shape,
					index + 1
				)
			end
		end

		return output
	end

	return initializerFunc(shape, 1)
end

function initializersModule.constant(shape, args)
	local function initializerFunc(shape, value, index)
		local output = {}

		if index == #shape then
			for a = 1, shape[index] do
				output[a] = value
			end
		else
			for a = 1, shape[index] do
				output[a] = initializerFunc(
					shape,
					value,
					index + 1
				)
			end
		end

		return output
	end

	return initializerFunc(shape, args.value, 1)
end

function initializersModule.uniformRandom(shape, args)
	local function initializerFunc(shape, lowerLimit, upperLimit, index)
		local output = {}

		if index == #shape then
			for a = 1, shape[index] do
				output[a] = lowerLimit + math.random() * (upperLimit - lowerLimit)
			end
		else
			for a = 1, shape[index] do
				output[a] = initializerFunc(
					shape,
					lowerLimit,
					upperLimit,
					index + 1
				)
			end
		end

		return output
	end

	return initializerFunc(shape, args.lowerLimit, args.upperLimit, 1)
end

function initializersModule.normalRandom(shape, args)
	local function initializerFunc(shape, mean, sd, index)
		local output = {}

		if index == #shape then
			for a = 1, shape[index] do
				output[a] = mean + math.sqrt(-2 * math.log(math.random())) * math.cos(2 * math.pi * math.random()) * sd
			end
		else
			for a = 1, shape[index] do
				output[a] = initializerFunc(
					shape,
					mean,
					sd,
					index + 1
				)
			end
		end

		return output
	end

	return initializerFunc(shape, args.mean, args.sd, 1)
end

function initializersModule.uniformXavier(shape, args)
	local function initializerFunc(shape, limit, index)
		local output = {}

		if index == #shape then
			for a = 1, shape[index] do
				output[a] = -limit + math.random() * (limit + limit)
			end
		else
			for a = 1, shape[index] do
				output[a] = initializerFunc(
					shape,
					limit,
					index + 1
				)
			end
		end

		return output
	end

	return initializerFunc(shape, math.sqrt(6 / (args.inputs + args.outputs)), 1)
end

function initializersModule.normalXavier(shape, args)
	local function initializerFunc(shape, sd, index)
		local output = {}

		if index == #shape then
			for a = 1, shape[index] do
				output[a] = math.sqrt(-2 * math.log(math.random())) * math.cos(2 * math.pi * math.random()) * sd
			end
		else
			for a = 1, shape[index] do
				output[a] = initializerFunc(
					shape,
					sd,
					index + 1
				)
			end
		end

		return output
	end

	return initializerFunc(shape, math.sqrt(6 / (args.inputs + args.outputs)), 1)
end

function initializersModule.uniformHe(shape, args)
	local function initializerFunc(shape, limit, index)
		local output = {}

		if index == #shape then
			for a = 1, shape[index] do
				output[a] = -limit + math.random() * (limit + limit)
			end
		else
			for a = 1, shape[index] do
				output[a] = initializersModule.uniformHe(
					shape,
					limit,
					index + 1
				)
			end
		end

		return output
	end

	return initializerFunc(shape, math.sqrt(2 / args.inputs), 1)
end

function initializersModule.normalHe(shape, args)
	local function initializerFunc(shape, sd, index)
		local output = {}

		if index == #shape then
			for a = 1, shape[index] do
				output[a] = math.sqrt(-2 * math.log(math.random())) * math.cos(2 * math.pi * math.random()) * sd
			end
		else
			for a = 1, shape[index] do
				output[a] = initializerFunc(
					shape,
					sd,
					index + 1
				)
			end
		end

		return output
	end

	return initializerFunc(shape, math.sqrt(2 / args.inputs), 1)
end

return initializersModule
