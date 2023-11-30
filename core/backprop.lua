--[[
	https://github.com/mochji/synapsea
	core/backprop.lua

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

local errorModule    = require("core.layers.error")
local gradientModule = require("core.layers.gradient")

local backPropModule = {
	outputError,
	gradientDescent = {
		stochastic,
		batch
	}
}

function backPropModule.outputError(output, expectedOutput, activation, alpha)
	local outputError = {}

	local activation = activationsModule[activation]

	for a = 1, #output do
		if canindex(output[a]) then
			outputError[a] = backPropModule.outputError(output[a], expectedOutput[a], activation)
		else
			outputError[a] = (output[a] - expectedOutput[a]) * activation(output[a], true, alpha)
		end
	end

	return outputError
end

function backPropModule.gradientDescent.stochastic(model, dataset, args)
end

function backPropModule.gradientDescent.batch(model, dataset, args)
end

return backPropModule