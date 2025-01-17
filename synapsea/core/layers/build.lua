--[[
	https://github.com/mochji/synapsea
	core/layers/build.lua

	Synapsea, a simple yet powerful machine learning framework for Lua.
	Copyright (C) 2024 mochji

	MIT License
]]--

local buildModule = {
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
	dropout
}

function buildModule.dense(layerConfig)
	assert(
		layerConfig.outputSize,
		"expected argument 'outputSize' to 'dense'"
	)

	local defaults = {
		activation       = "linear",
		useBias          = false,

		weightsInit      = "zeros",
		biasInit         = "zeros",

		weightsTrainable = false,
		biasTrainable    = false
	}

	layerConfig = setmetatable(layerConfig, {__index = defaults})

	local layer, parameterBuild = {
		config = {
			activation = layerConfig.activation,
			outputSize = layerConfig.outputSize
		},
		parameters = {
			alpha = layerConfig.alpha
		},
		trainable   = {},
		initializer = {},
		inputShape  = layerConfig.inputShape,
		outputShape = {layerConfig.outputSize}
	}, {}

	-- Weights

	layer.initializer.weights = {
		initializer = layerConfig.weightsInit,
		args        = layerConfig.weightsInitArgs
	}

	layer.trainable.weights =
		layerConfig.weightsTrainable and true or false

	parameterBuild.weights = {
		layerConfig.inputShape[1],
		layerConfig.outputSize
	}

	-- Biases

	if layerConfig.useBias then
		layer.initializer.bias = {
			initializer = layerConfig.biasInit,
			args        = layerConfig.biasInitArgs
		}

		layer.trainable.bias =
			layerConfig.biasTrainable and true or false

		layer.parameters.bias = 0
	end

	return layer, parameterBuild
end

function buildModule.averagePooling1D(layerConfig)
	local defaults = {
		kernel   = {1},
		stride   = {1},
		dilation = {1}
	}

	layerConfig = setmetatable(layerConfig, {__index = defaults})

	local layer = {
		config = {
			kernel   = layerConfig.kernel,
			stride   = layerConfig.stride,
			dilation = layerConfig.dilation
		},
		inputShape = layerConfig.inputShape
	}

	if layerConfig.inputShape[2] then
		layer.outputShape = {
			layerConfig.inputShape[1],
			math.floor((layerConfig.inputShape[2] - layerConfig.kernel[1]) / layerConfig.stride[1]) + 1
		}
	else
		layer.outputShape = {
			math.floor((layerConfig.inputShape[1] - layerConfig.kernel[1]) / layerConfig.stride[1]) + 1
		}
	end

	return layer
end

function buildModule.averagePooling2D(layerConfig)
	local defaults = {
		kernel   = {1, 1},
		stride   = {1, 1},
		dilation = {1, 1}
	}

	layerConfig = setmetatable(layerConfig, {__index = defaults})

	local layer = {
		config = {
			kernel   = layerConfig.kernel,
			stride   = layerConfig.stride,
			dilation = layerConfig.dilation
		},
		inputShape = layerConfig.inputShape
	}

	if layerConfig.inputShape[3] then
		layer.outputShape = {
			layerConfig.inputShape[1],
			math.floor((layerConfig.inputShape[2] - layerConfig.kernel[1]) / layerConfig.stride[1]) + 1,
			math.floor((layerConfig.inputShape[3] - layerConfig.kernel[2]) / layerConfig.stride[2]) + 1
		}
	else
		layer.outputShape = {
			math.floor((layerConfig.inputShape[1] - layerConfig.kernel[1]) / layerConfig.stride[1]) + 1,
			math.floor((layerConfig.inputShape[2] - layerConfig.kernel[2]) / layerConfig.stride[2]) + 1
		}
	end

	return layer
end

function buildModule.averagePooling3D(layerConfig)
	local defaults = {
		kernel   = {1, 1, 1},
		stride   = {1, 1, 1},
		dilation = {1, 1, 1}
	}

	layerConfig = setmetatable(layerConfig, {__index = defaults})

	local layer = {
		config = {
			kernel   = layerConfig.kernel,
			stride   = layerConfig.stride,
			dilation = layerConfig.dilation
		},
		inputShape = layerConfig.inputShape
	}

	if layerConfig.inputShape[4] then
		layer.outputShape = {
			layerConfig.inputShape[1],
			math.floor((layerConfig.inputShape[2] - layerConfig.kernel[1]) / layerConfig.stride[1]) + 1,
			math.floor((layerConfig.inputShape[3] - layerConfig.kernel[2]) / layerConfig.stride[2]) + 1,
			math.floor((layerConfig.inputShape[4] - layerConfig.kernel[3]) / layerConfig.stride[3]) + 1
		}
	else
		layer.outputShape = {
			math.floor((layerConfig.inputShape[1] - layerConfig.kernel[1]) / layerConfig.stride[1]) + 1,
			math.floor((layerConfig.inputShape[2] - layerConfig.kernel[2]) / layerConfig.stride[2]) + 1,
			math.floor((layerConfig.inputShape[3] - layerConfig.kernel[3]) / layerConfig.stride[3]) + 1
		}
	end

	return layer
end

buildModule.maxPooling1D = buildModule.averagePooling1D
buildModule.maxPooling2D = buildModule.averagePooling2D
buildModule.maxPooling3D = buildModule.averagePooling3D

buildModule.sumPooling1D = buildModule.averagePooling1D
buildModule.sumPooling2D = buildModule.averagePooling2D
buildModule.sumPooling3D = buildModule.averagePooling3D

function buildModule.upSample1D(layerConfig)
	local defaults = {
		kernel = {2}
	}

	layerConfig = setmetatable(layerConfig, {__index = defaults})

	local layer = {
		config = {
			kernel = layerConfig.kernel
		},
		inputShape = layerConfig.inputShape
	}

	if layerConfig.inputShape[2] then
		layer.outputShape = {
			layerConfig.inputShape[1],
			math.floor((layerConfig.inputShape[2] - layerConfig.kernel[1]) / layerConfig.stride[1]) + 1
		}
	else
		layer.outputShape = {
			math.floor((layerConfig.inputShape[1] - layerConfig.kernel[1]) / layerConfig.stride[1]) + 1
		}
	end

	return layer
end

function buildModule.upSample2D(layerConfig)
	local defaults = {
		kernel = {2, 2}
	}

	layerConfig = setmetatable(layerConfig, {__index = defaults})

	local layer = {
		config = {
			kernel = layerConfig.kernel
		},
		inputShape = layerConfig.inputShape
	}

	if layerConfig.inputShape[3] then
		layer.outputShape = {
			layerConfig.inputShape[1],
			layerConfig.inputShape[2] * layerConfig.kernel[1], layerConfig.inputShape[3] * layerConfig.kernel[2]
		}
	else
		layer.outputShape = {
			layerConfig.inputShape[1] * layerConfig.kernel[1], layerConfig.inputShape[2] * layerConfig.kernel[2]
		}
	end

	return layer
end

function buildModule.upSample3D(layerConfig)
	local defaults = {
		kernel = {2, 2}
	}

	layerConfig = setmetatable(layerConfig, {__index = defaults})

	local layer = {
		config = {
			kernel = layerConfig.kernel
		},
		inputShape = layerConfig.inputShape
	}

	if layerConfig.inputShape[4] then
		layer.outputShape = {
			layerConfig.inputShape[1],
			layerConfig.inputShape[2] * layerConfig.kernel[1],
			layerConfig.inputShape[3] * layerConfig.kernel[2],
			layerConfig.inputShape[4] * layerConfig.kernel[3]
		}
	else
		layer.outputShape = {
			layerConfig.inputShape[1] * layerConfig.kernel[1],
			layerConfig.inputShape[2] * layerConfig.kernel[2],
			layerConfig.inputShape[3] * layerConfig.kernel[3]
		}
	end

	return layer
end

function buildModule.zeroPad1D(layerConfig)
	local defaults = {
		paddingAmount = {1}
	}

	layerConfig = setmetatable(layerConfig, {__index = defaults})

	local layer = {
		config = {
			paddingAmount = layerConfig.paddingAmount
		},
		inputShape = layerConfig.inputShape
	}

	if layerConfig.inputShape[2] then
		layer.outputShape = {
			layerConfig.inputShape[1],
			layerConfig.inputShape[2] + layerConfig.paddingAmount[1] * 2
		}
	else
		layer.outputShape = {
			layerConfig.inputShape[1] + layerConfig.paddingAmount[1] * 2
		}
	end

	return layer
end

function buildModule.zeroPad2D(layerConfig)
	local defaults = {
		paddingAmount = {1, 1}
	}

	layerConfig = setmetatable(layerConfig, {__index = defaults})

	local layer = {
		config = {
			paddingAmount = layerConfig.paddingAmount
		},
		inputShape = layerConfig.inputShape
	}

	if layerConfig.inputShape[3] then
		layer.outputShape = {
			layerConfig.inputShape[1],
			layerConfig.inputShape[2] + layerConfig.paddingAmount[1] * 2,
			layerConfig.inputShape[3] + layerConfig.paddingAmount[2] * 2
		}
	else
		layer.outputShape = {
			layerConfig.inputShape[1] + layerConfig.paddingAmount[1] * 2,
			layerConfig.inputShape[2] + layerConfig.paddingAmount[2] * 2
		}
	end

	return layer
end

function buildModule.zeroPad3D(layerConfig)
	local defaults = {
		paddingAmount = {1, 1, 1}
	}

	layerConfig = setmetatable(layerConfig, {__index = defaults})

	local layer = {
		config = {
			paddingAmount = layerConfig.paddingAmount
		},
		inputShape = layerConfig.inputShape
	}

	if layerConfig.inputShape[4] then
		layer.outputShape = {
			layerConfig.inputShape[1],
			layerConfig.inputShape[2] + layerConfig.paddingAmount[1] * 2,
			layerConfig.inputShape[3] + layerConfig.paddingAmount[2] * 2,
			layerConfig.inputShape[4] + layerConfig.paddingAmount[3] * 2
		}
	else
		layer.outputShape = {
			layerConfig.inputShape[1] + layerConfig.paddingAmount[1] * 2,
			layerConfig.inputShape[2] + layerConfig.paddingAmount[2] * 2,
			layerConfig.inputShape[3] + layerConfig.paddingAmount[3] * 2
		}
	end

	return layer
end

function buildModule.crop1D(layerConfig)
	assert(
		layerConfig.outputShape,
		"expected argument 'outputShape' to 'crop1D'"
	)

	assert(
		layerConfig.start,
		"expected argument 'start' to 'crop1D'"
	)

	local layer = {
		config = {
			start       = layerConfig.start,
			outputShape = layerConfig.outputShape
		},
		inputShape = layerConfig.inputShape
	}

	if layerConfig.inputShape[2] then
		layer.outputShape = {
			layerConfig.inputShape[1],
			layerConfig.outputShape[1]
		}
	else
		layer.outputShape = layerConfig.outputShape
	end

	return layer
end

function buildModule.crop2D(layerConfig)
	assert(
		layerConfig.outputShape,
		"expected argument 'outputShape' to 'crop2D'"
	)

	assert(
		layerConfig.start,
		"expected argument 'start' to 'crop2D'"
	)

	local layer = {
		config = {
			start       = layerConfig.start,
			outputShape = layerConfig.outputShape
		},
		inputShape = layerConfig.inputShape
	}

	if layerConfig.inputShape[2] then
		layer.outputShape = {
			layerConfig.inputShape[1],
			layerConfig.outputShape[1],
			layerConfig.outputShape[2]
		}
	else
		layer.outputShape = layerConfig.outputShape
	end

	return layer
end

function buildModule.crop3D(layerConfig)
	assert(
		layerConfig.outputShape,
		"expected argument 'outputShape' to 'crop3D'"
	)

	assert(
		layerConfig.start,
		"expected argument 'start' to 'crop3D'"
	)

	local layer = {
		config = {
			start       = layerConfig.start,
			outputShape = layerConfig.outputShape
		},
		inputShape = layerConfig.inputShape
	}

	if layerConfig.inputShape[2] then
		layer.outputShape = {
			layerConfig.inputShape[1],
			layerConfig.outputShape[1],
			layerConfig.outputShape[2],
			layerConfig.outputShape[3]
		}
	else
		layer.outputShape = layerConfig.outputShape
	end

	return layer
end

function buildModule.convolutional1D(layerConfig)
	local defaults = {
		activation      = "linear",
		useBias         = false,

		kernel          = {2},
		stride          = {1},
		dilation        = {1},
		filters         = 1,

		filterInit      = "zeros",
		biasesInit      = "zeros",

		filterTrainable = false,
		biasesTrainable = false
	}

	layerConfig = setmetatable(layerConfig, {__index = defaults})

	local layer, parameterBuild = {
		config = {
			activation = layerConfig.activation,
			stride     = layerConfig.stride,
			dilation   = layerConfig.dilation,
			filters    = layerConfig.filters
		},
		parameters = {
			alpha = layerConfig.alpha
		},
		trainable   = {},
		initializer = {},
		inputShape  = layerConfig.inputShape
	}, {}

	-- Output shape

	if layerConfig.inputShape[2] then
		layer.outputShape = {
			math.floor((layerConfig.inputShape[2] - layerConfig.kernel[1]) / layerConfig.stride[1]) + 1
		}
	else
		layer.outputShape = {
			math.floor((layerConfig.inputShape[1] - layerConfig.kernel[1]) / layerConfig.stride[1]) + 1
		}
	end

	if layerConfig.filters > 1 then
		table.insert(layer.outputShape, 1, layerConfig.filters)
	end

	-- Filter

	layer.initializer.filter = {
		initializer = layerConfig.filterInit,
		args        = layerConfig.filterInitArgs
	}

	layer.trainable.filter =
		layerConfig.filterTrainable and true or false

	parameterBuild.filter = {
		layerConfig.filters,
		layerConfig.kernel[1]
	}

	-- Biases

	if layerConfig.useBias then
		layer.initializer.biases = {
			initializer = layerConfig.biasesInit,
			args        = layerConfig.biasesInitArgs
		}

		layer.trainable.biases =
			layerConfig.filterTrainable and true or false

		parameterBuild.biases = {
			layerConfig.filters,
			layer.outputShape[1]
		}
	end

	return layer, parameterBuild
end

function buildModule.convolutional2D(layerConfig)
	local defaults = {
		activation      = "linear",
		useBias         = false,

		kernel          = {2, 2},
		stride          = {1, 1},
		dilation        = {1, 1},
		filters         = 1,

		filterInit      = "zeros",
		biasesInit      = "zeros",

		filterTrainable = false,
		biasesTrainable = false
	}

	layerConfig = setmetatable(layerConfig, {__index = defaults})

	local layer, parameterBuild = {
		config = {
			activation = layerConfig.activation,
			stride     = layerConfig.stride,
			dilation   = layerConfig.dilation,
			filters    = layerConfig.filters
		},
		parameters = {
			alpha = layerConfig.alpha
		},
		trainable   = {},
		initializer = {},
		inputShape  = layerConfig.inputShape
	}, {}

	-- Output shape

	if layerConfig.inputShape[3] then
		layer.outputShape = {
			math.floor((layerConfig.inputShape[2] - layerConfig.kernel[1]) / layerConfig.stride[1]) + 1,
			math.floor((layerConfig.inputShape[3] - layerConfig.kernel[2]) / layerConfig.stride[2]) + 1
		}
	else
		layer.outputShape = {
			math.floor((layerConfig.inputShape[1] - layerConfig.kernel[1]) / layerConfig.stride[1]) + 1,
			math.floor((layerConfig.inputShape[2] - layerConfig.kernel[2]) / layerConfig.stride[2]) + 1
		}
	end

	if layerConfig.filters > 1 then
		table.insert(layer.outputShape, 1, layerConfig.filters)
	end

	-- Filter

	layer.initializer.filter = {
		initializer = layerConfig.filterInit,
		args        = layerConfig.filterInitArgs
	}

	layer.trainable.filter =
		layerConfig.filterTrainable and true or false

	parameterBuild.filter = {
		layerConfig.filters,
		layerConfig.kernel[1],
		layerConfig.kernel[2]
	}

	-- Biases

	if layerConfig.useBias then
		layer.initializer.biases = {
			initializer = layerConfig.biasesInit,
			args        = layerConfig.biasesInitArgs
		}

		layer.trainable.biases =
			layerConfig.filterTrainable and true or false

		parameterBuild.biases = {
			layerConfig.filters,
			layer.outputShape[1],
			layer.outputShape[2]
		}
	end

	return layer, parameterBuild
end

function buildModule.convolutional3D(layerConfig)
	local defaults = {
		activation      = "linear",
		useBias         = false,

		kernel          = {2, 2, 2},
		stride          = {1, 1, 1},
		dilation        = {1, 1, 1},
		filters         = 1,

		filterInit      = "zeros",
		biasesInit      = "zeros",

		filterTrainable = false,
		biasesTrainable = false
	}

	layerConfig = setmetatable(layerConfig, {__index = defaults})

	local layer, parameterBuild = {
		config = {
			activation = layerConfig.activation,
			stride     = layerConfig.stride,
			dilation   = layerConfig.dilation,
			filters    = layerConfig.filters
		},
		parameters = {
			alpha = layerConfig.alpha
		},
		trainable   = {},
		initializer = {},
		inputShape  = layerConfig.inputShape
	}, {}

	-- Output shape

	if layerConfig.inputShape[4] then
		layer.outputShape = {
			math.floor((layerConfig.inputShape[2] - layerConfig.kernel[1]) / layerConfig.stride[1]) + 1,
			math.floor((layerConfig.inputShape[3] - layerConfig.kernel[2]) / layerConfig.stride[2]) + 1,
			math.floor((layerConfig.inputShape[4] - layerConfig.kernel[3]) / layerConfig.stride[3]) + 1
		}
	else
		layer.outputShape = {
			math.floor((layerConfig.inputShape[1] - layerConfig.kernel[1]) / layerConfig.stride[1]) + 1,
			math.floor((layerConfig.inputShape[2] - layerConfig.kernel[2]) / layerConfig.stride[2]) + 1,
			math.floor((layerConfig.inputShape[3] - layerConfig.kernel[3]) / layerConfig.stride[3]) + 1
		}
	end

	if layerConfig.filters > 1 then
		table.insert(layer.outputShape, 1, layerConfig.filters)
	end

	-- Filter

	layer.initializer.filter = {
		initializer = layerConfig.filterInit,
		args        = layerConfig.filterInitArgs
	}

	layer.trainable.filter =
		layerConfig.filterTrainable and true or false

	parameterBuild.filter = {
		layerConfig.filters,
		layerConfig.kernel[1],
		layerConfig.kernel[2],
		layerConfig.kernel[3]
	}

	-- Biases

	if layerConfig.useBias then
		layer.initializer.biases = {
			initializer = layerConfig.biasesInit,
			args  = layerConfig.biasesInitArgs
		}

		layer.trainable.biases =
			layerConfig.biasesTrainable and true or false

		parameterBuild.biases = {
			layerConfig.filters,
			layer.outputShape[1],
			layer.outputShape[2],
			layer.outputShape[3]
		}
	end

	return layer, parameterBuild
end

function buildModule.convolutionalTranspose1D(layerConfig)
	local defaults = {
		activation      = "linear",
		useBias         = false,

		kernel          = {2},
		stride          = {1},
		dilation        = {1},
		paddingAmount   = {1},
		filters         = 1,

		filterInit      = "zeros",
		biasesInit      = "zeros",

		filterTrainable = false,
		biasesTrainable = false
	}

	layerConfig = setmetatable(layerConfig, {__index = defaults})

	local layer, parameterBuild = {
		config = {
			activation = layerConfig.activation,
			stride     = layerConfig.stride,
			dilation   = layerConfig.dilation,
			filters    = layerConfig.filters
		},
		parameters = {
			alpha = layerConfig.alpha
		},
		trainable   = {},
		initializer = {},
		inputShape  = layerConfig.inputShape
	}, {}

	-- Output shape

	if layerConfig.inputShape[2] then
		layer.outputShape = {
			math.floor(
				((layerConfig.inputShape[2] + layerConfig.paddingAmount[1]) - layerConfig.kernel[1])
				/ layerConfig.stride[1]
			) + 1
		}
	else
		layer.outputShape = {
			math.floor(
				((layerConfig.inputShape[1] + layerConfig.paddingAmount[1]) - layerConfig.kernel[1])
				/ layerConfig.stride[1]
			) + 1
		}
	end

	if layerConfig.filters > 1 then
		table.insert(layer.outputShape, 1, layerConfig.filters)
	end

	-- Filter

	layer.initializer.filter = {
		initializer = layerConfig.filterInit,
		args        = layerConfig.filterInitArgs
	}

	layer.trainable.filter =
		layerConfig.filterTrainable and true or false

	parameterBuild.filter = {
		layerConfig.filters,
		layerConfig.kernel[1]
	}

	-- Biases

	if layerConfig.useBias then
		layer.initializer.biases = {
			initializer = layerConfig.biasesInit,
			args        = layerConfig.biasesInitArgs
		}

		layer.trainable.biases =
			layerConfig.filterTrainable and true or false

		parameterBuild.biases = {
			layerConfig.filters,
			layer.outputShape[1]
		}
	end

	return layer, parameterBuild
end

function buildModule.convolutionalTranspose2D(layerConfig)
	local defaults = {
		activation      = "linear",
		useBias         = false,

		kernel          = {2, 2},
		stride          = {1, 1},
		dilation        = {1, 1},
		paddingAmount   = {1, 1},
		filters         = 1,

		filterInit      = "zeros",
		biasesInit      = "zeros",

		filterTrainable = false,
		biasesTrainable = false
	}

	layerConfig = setmetatable(layerConfig, {__index = defaults})

	local layer, parameterBuild = {
		config = {
			activation = layerConfig.activation,
			stride     = layerConfig.stride,
			dilation   = layerConfig.dilation,
			filters    = layerConfig.filters
		},
		parameters = {
			alpha = layerConfig.alpha
		},
		trainable   = {},
		initializer = {},
		inputShape  = layerConfig.inputShape
	}, {}

	-- Output shape

	if layerConfig.inputShape[2] then
		layer.outputShape = {
			math.floor(
				((layerConfig.inputShape[2] + layerConfig.paddingAmount[1]) - layerConfig.kernel[1])
				/ layerConfig.stride[1]
			) + 1,
			math.floor(
				((layerConfig.inputShape[3] + layerConfig.paddingAmount[2]) - layerConfig.kernel[2])
				/ layerConfig.stride[2]
			) + 1
		}
	else
		layer.outputShape = {
			math.floor(
				((layerConfig.inputShape[1] + layerConfig.paddingAmount[1]) - layerConfig.kernel[1])
				/ layerConfig.stride[1]
			) + 1,
			math.floor(
				((layerConfig.inputShape[2] + layerConfig.paddingAmount[2]) - layerConfig.kernel[2])
				/ layerConfig.stride[2]
			) + 1
		}
	end

	if layerConfig.filters > 1 then
		table.insert(layer.outputShape, 1, layerConfig.filters)
	end

	-- Filter

	layer.initializer.filter = {
		initializer = layerConfig.filterInit,
		args        = layerConfig.filterInitArgs
	}

	layer.trainable.filter =
		layerConfig.filterTrainable and true or false

	parameterBuild.filter = {
		layerConfig.filters,
		layerConfig.kernel[1],
		layerConfig.kernel[2]
	}

	-- Biases

	if layerConfig.useBias then
		layer.initializer.biases = {
			initializer = layerConfig.biasesInit,
			args        = layerConfig.biasesInitArgs
		}

		layer.trainable.biases =
			layerConfig.filterTrainable and layerConfig.useBias or false

		parameterBuild.filter = {
			layerConfig.filters,
			layer.outputShape[1],
			layer.outputShape[2]
		}
	end

	return layer, parameterBuild
end

function buildModule.convolutionalTranspose3D(layerConfig)
	local defaults = {
		activation      = "linear",
		useBias         = false,

		kernel          = {2, 2, 2},
		stride          = {1, 1, 1},
		dilation        = {1, 1, 1},
		paddingAmount   = {1, 1, 1},
		filters         = 1,

		filterInit      = "zeros",
		biasesInit      = "zeros",

		filterTrainable = false,
		biasesTrainable = false
	}

	layerConfig = setmetatable(layerConfig, {__index = defaults})

	local layer, parameterBuild = {
		config = {
			activation = layerConfig.activation,
			stride     = layerConfig.stride,
			dilation   = layerConfig.dilation,
			filters    = layerConfig.filters
		},
		parameters = {
			alpha = layerConfig.alpha
		},
		trainable   = {},
		initializer = {},
		inputShape  = layerConfig.inputShape
	}, {}

	-- Output shape

	if layerConfig.inputShape[2] then
		layer.outputShape = {
			math.floor(
				((layerConfig.inputShape[2] + layerConfig.paddingAmount[1]) - layerConfig.kernel[1])
				/ layerConfig.stride[1]
			) + 1,
			math.floor(
				((layerConfig.inputShape[3] + layerConfig.paddingAmount[2]) - layerConfig.kernel[2])
				/ layerConfig.stride[2]
			) + 1,
			math.floor(
				((layerConfig.inputShape[4] + layerConfig.paddingAmount[3]) - layerConfig.kernel[3])
				/ layerConfig.stride[3]
			) + 1
		}
	else
		layer.outputShape = {
			math.floor(
				((layerConfig.inputShape[1] + layerConfig.paddingAmount[1]) - layerConfig.kernel[1])
				/ layerConfig.stride[1]
			) + 1,
			math.floor(
				((layerConfig.inputShape[2] + layerConfig.paddingAmount[2]) - layerConfig.kernel[2])
				/ layerConfig.stride[2]
			) + 1,
			math.floor(
				((layerConfig.inputShape[3] + layerConfig.paddingAmount[3]) - layerConfig.kernel[3])
				/ layerConfig.stride[3]
			) + 1
		}
	end

	if layerConfig.filters > 1 then
		table.insert(layer.outputShape, 1, layerConfig.filters)
	end

	-- Filters

	layer.initializer.filter = {
		initializer = layerConfig.filterInit,
		args        = layerConfig.filterInitArgs
	}

	layer.trainable.filter =
		layerConfig.filterTrainable and true or false

	parameterBuild.filter = {
		layerConfig.filters,
		layerConfig.kernel[1],
		layerConfig.kernel[2],
		layerConfig.kernel[3]
	}

	-- Biases

	if layerConfig.useBias then
		layer.initializer.biases = {
			initializer = layerConfig.biasesInit,
			args        = layerConfig.biasesInitArgs
		}

		layer.trainable.biases =
			layerConfig.filterTrainable and layerConfig.useBias or false

		parameterBuild.biases = {
			layerConfig.filters,
			layer.outputShape[1],
			layer.outputShape[2],
			layer.outputShape[3]
		}
	end

	return layer, parameterBuild
end

function buildModule.flatten(layerConfig)
	local outputShape = 1

	for a = 1, #layerConfig.inputShape do
		outputShape = outputShape * layerConfig.inputShape[a]
	end

	return {
		inputShape = layerConfig.inputShape,
		outputShape = {outputShape}
	}
end

function buildModule.reshape(layerConfig)
	assert(
		layerConfig.shape,
		"expected argument 'shape' to 'reshape'"
	)

	local totalIn, totalOut = 1, 1

	for a = 1, #layerConfig.inputShape do
		totalIn = totalIn * layerConfig.inputShape[a]
	end

	for a = 1, #layerConfig.shape do
		totalOut = totalOut * layerConfig.shape[a]
	end

	assert(
		totalIn == totalOut,
		"input cannot be reshaped to specified shape"
	)

	return {
		config = {
			shape = layerConfig.shape
		},
		inputShape  = layerConfig.inputShape,
		outputShape = layerConfig.shape
	}
end

function buildModule.add1D(layerConfig)
	local defaults = {
		biasesInit = "zeros"
	}

	layerConfig = setmetatable(layerConfig, {__index = defaults})

	local layer, parameterBuild = {
		parameters  = {},
		trainable   = {},
		initializer = {},
		inputShape  = layerConfig.inputShape,
		outputShape = layerConfig.inputShape
	}, {}

	-- Initializers

	layer.initializer.biases = {
		initializer = layerConfig.biasesInit,
		args        = layerConfig.biasesInitArgs
	}

	-- Trainable

	layer.trainable.biases =
		layerConfig.biasesTrainable and true or false

	-- Parameters

	parameterBuild.biases = {layerConfig.inputShape}

	return layer, parameterBuild
end

buildModule.add2D = buildModule.add1D
buildModule.add3D = buildModule.add1D

buildModule.subtract1D = buildModule.add1D
buildModule.subtract2D = buildModule.add1D
buildModule.subtract3D = buildModule.add1D

function buildModule.multiply1D(layerConfig)
	local defaults = {
		weightsInit = "zeros"
	}

	layerConfig = setmetatable(layerConfig, {__index = defaults})

	local layer, parameterBuild = {
		parameters  = {},
		trainable   = {},
		initializer = {},
		inputShape  = layerConfig.inputShape,
		outputShape = layerConfig.inputShape
	}, {}

	-- Initializers

	layer.initializer.weights = {
		initializer = layerConfig.weightsInit,
		args        = layerConfig.weightsInitArgs
	}

	-- Trainable

	layer.trainable.weights =
		layerConfig.weightsTrainable and true or false

	-- Parameters

	parameterBuild.weights = {layerConfig.inputShape}

	return layer, parameterBuild
end

buildModule.multiply2D = buildModule.multiply1D
buildModule.multiply3D = buildModule.multiply1D

buildModule.divide1D = buildModule.multiply1D
buildModule.divide2D = buildModule.multiply1D
buildModule.divide3D = buildModule.multiply1D

function buildModule.softmax(layerConfig)
	return {
		inputShape  = layerConfig.inputShape,
		outputShape = layerConfig.inputShape
	}
end

function buildModule.activate(layerConfig)
	assert(
		layerConfig.activation,
		"expected argument 'activation' to 'activate'"
	)

	return {
		config = {
			activation = layerConfig.activation,
			derivative =
				layerConfig.derivative and true or false
		},
		parameters = {
			alpha = layerConfig.alpha
		},
		inputShape  = layerConfig.inputShape,
		outputShape = layerConfig.inputShape
	}
end

function buildModule.dropout(layerConfig)
	assert(
		layerConfig.rate,
		"expected argument 'rate' to 'dropout'"
	)

	return {
		config = {
			rate = layerConfig.rate
		},
		inputShape  = layerConfig.inputShape,
		outputShape = layerConfig.inputShape
	}
end

return buildModule
