module KeemenaLMMetalExt

using KeemenaLM
import Metal

functional() = Metal.functional()
to_device(value) = Metal.mtl(value)
is_device_array(value) = value isa Metal.MtlArray

end # module KeemenaLMMetalExt
