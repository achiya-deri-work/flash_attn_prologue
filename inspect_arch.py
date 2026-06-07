import cutlass

dsl = cutlass.base_dsl.BaseDSL._get_dsl()
print("Arch enum:", dsl.get_arch_enum())
print("Arch name:", dsl.get_arch_name())
