all: r_0 r_1 r_2 r_3 r_4 r_5 r_6 r_7 r_8 r_9 r_10 r_11 r_12 r_13 r_14 r_15 r_16 r_17 r_18 r_19 r_20 r_21 r_22 r_23 r_24 r_25 r_26 r_27 r_28 r_29 r_30 r_31 r_32 r_33 r_34 r_35 r_36 r_37 r_38 r_39 r_40 r_41 r_42 r_43 r_44 r_45 r_46 r_47 r_48 r_49 r_50 r_51 r_52 r_53 r_54 r_55 r_56 r_57 r_58 r_59 r_60 r_61 r_62 r_63 r_64 r_65 r_66 r_67 r_68 r_69 r_70 r_71 r_72 r_73 r_74 r_75 r_76 r_77 r_78 r_79 r_80 r_81 r_82 r_83 r_84 r_85 r_86 r_87 r_88 r_89 r_90 r_91 r_92 r_93 r_94 r_95 r_96 r_97 r_98 r_99 r_100 r_101 r_102 r_103 r_104 r_105 r_106 r_107 r_108 r_109 r_110 r_111 r_112 r_113 r_114 r_115 r_116 r_117 r_118 r_119
	numba quicksort.py merge *.o

clean:
	rm -f *.o *.pickle *.so

run:
	python -c "import my_module; print(dir(my_module))"

r_0:
		numba quicksort.py emit-obj -f quicksort_1 -n quicksort_int64 -s "none(int64[:], int64, int64, int64[:], int64[:])" -o quicksort_int64.o

r_1:
		numba quicksort.py emit-obj -f quicksort_1 -n quicksort_int32 -s "none(int64[:], int64, int64, int64[:], int32[:])" -o quicksort_int32.o

r_2:
		numba quicksort.py emit-obj -f quicksort_1 -n quicksort_float64 -s "none(int64[:], int64, int64, int64[:], float64[:])" -o quicksort_float64.o

r_3:
		numba quicksort.py emit-obj -f quicksort_2 -n quicksort_int64_int64 -s "none(int64[:], int64, int64, int64[:], int64[:], int64[:])" -o quicksort_int64_int64.o

r_4:
		numba quicksort.py emit-obj -f quicksort_2 -n quicksort_int64_int32 -s "none(int64[:], int64, int64, int64[:], int64[:], int32[:])" -o quicksort_int64_int32.o

r_5:
		numba quicksort.py emit-obj -f quicksort_2 -n quicksort_int64_float64 -s "none(int64[:], int64, int64, int64[:], int64[:], float64[:])" -o quicksort_int64_float64.o

r_6:
		numba quicksort.py emit-obj -f quicksort_2 -n quicksort_int32_int64 -s "none(int64[:], int64, int64, int64[:], int32[:], int64[:])" -o quicksort_int32_int64.o

r_7:
		numba quicksort.py emit-obj -f quicksort_2 -n quicksort_int32_int32 -s "none(int64[:], int64, int64, int64[:], int32[:], int32[:])" -o quicksort_int32_int32.o

r_8:
		numba quicksort.py emit-obj -f quicksort_2 -n quicksort_int32_float64 -s "none(int64[:], int64, int64, int64[:], int32[:], float64[:])" -o quicksort_int32_float64.o

r_9:
		numba quicksort.py emit-obj -f quicksort_2 -n quicksort_float64_int64 -s "none(int64[:], int64, int64, int64[:], float64[:], int64[:])" -o quicksort_float64_int64.o

r_10:
		numba quicksort.py emit-obj -f quicksort_2 -n quicksort_float64_int32 -s "none(int64[:], int64, int64, int64[:], float64[:], int32[:])" -o quicksort_float64_int32.o

r_11:
		numba quicksort.py emit-obj -f quicksort_2 -n quicksort_float64_float64 -s "none(int64[:], int64, int64, int64[:], float64[:], float64[:])" -o quicksort_float64_float64.o

r_12:
		numba quicksort.py emit-obj -f quicksort_3 -n quicksort_int64_int64_int64 -s "none(int64[:], int64, int64, int64[:], int64[:], int64[:], int64[:])" -o quicksort_int64_int64_int64.o

r_13:
		numba quicksort.py emit-obj -f quicksort_3 -n quicksort_int64_int64_int32 -s "none(int64[:], int64, int64, int64[:], int64[:], int64[:], int32[:])" -o quicksort_int64_int64_int32.o

r_14:
		numba quicksort.py emit-obj -f quicksort_3 -n quicksort_int64_int64_float64 -s "none(int64[:], int64, int64, int64[:], int64[:], int64[:], float64[:])" -o quicksort_int64_int64_float64.o

r_15:
		numba quicksort.py emit-obj -f quicksort_3 -n quicksort_int64_int32_int64 -s "none(int64[:], int64, int64, int64[:], int64[:], int32[:], int64[:])" -o quicksort_int64_int32_int64.o

r_16:
		numba quicksort.py emit-obj -f quicksort_3 -n quicksort_int64_int32_int32 -s "none(int64[:], int64, int64, int64[:], int64[:], int32[:], int32[:])" -o quicksort_int64_int32_int32.o

r_17:
		numba quicksort.py emit-obj -f quicksort_3 -n quicksort_int64_int32_float64 -s "none(int64[:], int64, int64, int64[:], int64[:], int32[:], float64[:])" -o quicksort_int64_int32_float64.o

r_18:
		numba quicksort.py emit-obj -f quicksort_3 -n quicksort_int64_float64_int64 -s "none(int64[:], int64, int64, int64[:], int64[:], float64[:], int64[:])" -o quicksort_int64_float64_int64.o

r_19:
		numba quicksort.py emit-obj -f quicksort_3 -n quicksort_int64_float64_int32 -s "none(int64[:], int64, int64, int64[:], int64[:], float64[:], int32[:])" -o quicksort_int64_float64_int32.o

r_20:
		numba quicksort.py emit-obj -f quicksort_3 -n quicksort_int64_float64_float64 -s "none(int64[:], int64, int64, int64[:], int64[:], float64[:], float64[:])" -o quicksort_int64_float64_float64.o

r_21:
		numba quicksort.py emit-obj -f quicksort_3 -n quicksort_int32_int64_int64 -s "none(int64[:], int64, int64, int64[:], int32[:], int64[:], int64[:])" -o quicksort_int32_int64_int64.o

r_22:
		numba quicksort.py emit-obj -f quicksort_3 -n quicksort_int32_int64_int32 -s "none(int64[:], int64, int64, int64[:], int32[:], int64[:], int32[:])" -o quicksort_int32_int64_int32.o

r_23:
		numba quicksort.py emit-obj -f quicksort_3 -n quicksort_int32_int64_float64 -s "none(int64[:], int64, int64, int64[:], int32[:], int64[:], float64[:])" -o quicksort_int32_int64_float64.o

r_24:
		numba quicksort.py emit-obj -f quicksort_3 -n quicksort_int32_int32_int64 -s "none(int64[:], int64, int64, int64[:], int32[:], int32[:], int64[:])" -o quicksort_int32_int32_int64.o

r_25:
		numba quicksort.py emit-obj -f quicksort_3 -n quicksort_int32_int32_int32 -s "none(int64[:], int64, int64, int64[:], int32[:], int32[:], int32[:])" -o quicksort_int32_int32_int32.o

r_26:
		numba quicksort.py emit-obj -f quicksort_3 -n quicksort_int32_int32_float64 -s "none(int64[:], int64, int64, int64[:], int32[:], int32[:], float64[:])" -o quicksort_int32_int32_float64.o

r_27:
		numba quicksort.py emit-obj -f quicksort_3 -n quicksort_int32_float64_int64 -s "none(int64[:], int64, int64, int64[:], int32[:], float64[:], int64[:])" -o quicksort_int32_float64_int64.o

r_28:
		numba quicksort.py emit-obj -f quicksort_3 -n quicksort_int32_float64_int32 -s "none(int64[:], int64, int64, int64[:], int32[:], float64[:], int32[:])" -o quicksort_int32_float64_int32.o

r_29:
		numba quicksort.py emit-obj -f quicksort_3 -n quicksort_int32_float64_float64 -s "none(int64[:], int64, int64, int64[:], int32[:], float64[:], float64[:])" -o quicksort_int32_float64_float64.o

r_30:
		numba quicksort.py emit-obj -f quicksort_3 -n quicksort_float64_int64_int64 -s "none(int64[:], int64, int64, int64[:], float64[:], int64[:], int64[:])" -o quicksort_float64_int64_int64.o

r_31:
		numba quicksort.py emit-obj -f quicksort_3 -n quicksort_float64_int64_int32 -s "none(int64[:], int64, int64, int64[:], float64[:], int64[:], int32[:])" -o quicksort_float64_int64_int32.o

r_32:
		numba quicksort.py emit-obj -f quicksort_3 -n quicksort_float64_int64_float64 -s "none(int64[:], int64, int64, int64[:], float64[:], int64[:], float64[:])" -o quicksort_float64_int64_float64.o

r_33:
		numba quicksort.py emit-obj -f quicksort_3 -n quicksort_float64_int32_int64 -s "none(int64[:], int64, int64, int64[:], float64[:], int32[:], int64[:])" -o quicksort_float64_int32_int64.o

r_34:
		numba quicksort.py emit-obj -f quicksort_3 -n quicksort_float64_int32_int32 -s "none(int64[:], int64, int64, int64[:], float64[:], int32[:], int32[:])" -o quicksort_float64_int32_int32.o

r_35:
		numba quicksort.py emit-obj -f quicksort_3 -n quicksort_float64_int32_float64 -s "none(int64[:], int64, int64, int64[:], float64[:], int32[:], float64[:])" -o quicksort_float64_int32_float64.o

r_36:
		numba quicksort.py emit-obj -f quicksort_3 -n quicksort_float64_float64_int64 -s "none(int64[:], int64, int64, int64[:], float64[:], float64[:], int64[:])" -o quicksort_float64_float64_int64.o

r_37:
		numba quicksort.py emit-obj -f quicksort_3 -n quicksort_float64_float64_int32 -s "none(int64[:], int64, int64, int64[:], float64[:], float64[:], int32[:])" -o quicksort_float64_float64_int32.o

r_38:
		numba quicksort.py emit-obj -f quicksort_3 -n quicksort_float64_float64_float64 -s "none(int64[:], int64, int64, int64[:], float64[:], float64[:], float64[:])" -o quicksort_float64_float64_float64.o

r_39:
		numba quicksort.py emit-obj -f quicksort_4 -n quicksort_int64_int64_int64_int64 -s "none(int64[:], int64, int64, int64[:], int64[:], int64[:], int64[:], int64[:])" -o quicksort_int64_int64_int64_int64.o

r_40:
		numba quicksort.py emit-obj -f quicksort_4 -n quicksort_int64_int64_int64_int32 -s "none(int64[:], int64, int64, int64[:], int64[:], int64[:], int64[:], int32[:])" -o quicksort_int64_int64_int64_int32.o

r_41:
		numba quicksort.py emit-obj -f quicksort_4 -n quicksort_int64_int64_int64_float64 -s "none(int64[:], int64, int64, int64[:], int64[:], int64[:], int64[:], float64[:])" -o quicksort_int64_int64_int64_float64.o

r_42:
		numba quicksort.py emit-obj -f quicksort_4 -n quicksort_int64_int64_int32_int64 -s "none(int64[:], int64, int64, int64[:], int64[:], int64[:], int32[:], int64[:])" -o quicksort_int64_int64_int32_int64.o

r_43:
		numba quicksort.py emit-obj -f quicksort_4 -n quicksort_int64_int64_int32_int32 -s "none(int64[:], int64, int64, int64[:], int64[:], int64[:], int32[:], int32[:])" -o quicksort_int64_int64_int32_int32.o

r_44:
		numba quicksort.py emit-obj -f quicksort_4 -n quicksort_int64_int64_int32_float64 -s "none(int64[:], int64, int64, int64[:], int64[:], int64[:], int32[:], float64[:])" -o quicksort_int64_int64_int32_float64.o

r_45:
		numba quicksort.py emit-obj -f quicksort_4 -n quicksort_int64_int64_float64_int64 -s "none(int64[:], int64, int64, int64[:], int64[:], int64[:], float64[:], int64[:])" -o quicksort_int64_int64_float64_int64.o

r_46:
		numba quicksort.py emit-obj -f quicksort_4 -n quicksort_int64_int64_float64_int32 -s "none(int64[:], int64, int64, int64[:], int64[:], int64[:], float64[:], int32[:])" -o quicksort_int64_int64_float64_int32.o

r_47:
		numba quicksort.py emit-obj -f quicksort_4 -n quicksort_int64_int64_float64_float64 -s "none(int64[:], int64, int64, int64[:], int64[:], int64[:], float64[:], float64[:])" -o quicksort_int64_int64_float64_float64.o

r_48:
		numba quicksort.py emit-obj -f quicksort_4 -n quicksort_int64_int32_int64_int64 -s "none(int64[:], int64, int64, int64[:], int64[:], int32[:], int64[:], int64[:])" -o quicksort_int64_int32_int64_int64.o

r_49:
		numba quicksort.py emit-obj -f quicksort_4 -n quicksort_int64_int32_int64_int32 -s "none(int64[:], int64, int64, int64[:], int64[:], int32[:], int64[:], int32[:])" -o quicksort_int64_int32_int64_int32.o

r_50:
		numba quicksort.py emit-obj -f quicksort_4 -n quicksort_int64_int32_int64_float64 -s "none(int64[:], int64, int64, int64[:], int64[:], int32[:], int64[:], float64[:])" -o quicksort_int64_int32_int64_float64.o

r_51:
		numba quicksort.py emit-obj -f quicksort_4 -n quicksort_int64_int32_int32_int64 -s "none(int64[:], int64, int64, int64[:], int64[:], int32[:], int32[:], int64[:])" -o quicksort_int64_int32_int32_int64.o

r_52:
		numba quicksort.py emit-obj -f quicksort_4 -n quicksort_int64_int32_int32_int32 -s "none(int64[:], int64, int64, int64[:], int64[:], int32[:], int32[:], int32[:])" -o quicksort_int64_int32_int32_int32.o

r_53:
		numba quicksort.py emit-obj -f quicksort_4 -n quicksort_int64_int32_int32_float64 -s "none(int64[:], int64, int64, int64[:], int64[:], int32[:], int32[:], float64[:])" -o quicksort_int64_int32_int32_float64.o

r_54:
		numba quicksort.py emit-obj -f quicksort_4 -n quicksort_int64_int32_float64_int64 -s "none(int64[:], int64, int64, int64[:], int64[:], int32[:], float64[:], int64[:])" -o quicksort_int64_int32_float64_int64.o

r_55:
		numba quicksort.py emit-obj -f quicksort_4 -n quicksort_int64_int32_float64_int32 -s "none(int64[:], int64, int64, int64[:], int64[:], int32[:], float64[:], int32[:])" -o quicksort_int64_int32_float64_int32.o

r_56:
		numba quicksort.py emit-obj -f quicksort_4 -n quicksort_int64_int32_float64_float64 -s "none(int64[:], int64, int64, int64[:], int64[:], int32[:], float64[:], float64[:])" -o quicksort_int64_int32_float64_float64.o

r_57:
		numba quicksort.py emit-obj -f quicksort_4 -n quicksort_int64_float64_int64_int64 -s "none(int64[:], int64, int64, int64[:], int64[:], float64[:], int64[:], int64[:])" -o quicksort_int64_float64_int64_int64.o

r_58:
		numba quicksort.py emit-obj -f quicksort_4 -n quicksort_int64_float64_int64_int32 -s "none(int64[:], int64, int64, int64[:], int64[:], float64[:], int64[:], int32[:])" -o quicksort_int64_float64_int64_int32.o

r_59:
		numba quicksort.py emit-obj -f quicksort_4 -n quicksort_int64_float64_int64_float64 -s "none(int64[:], int64, int64, int64[:], int64[:], float64[:], int64[:], float64[:])" -o quicksort_int64_float64_int64_float64.o

r_60:
		numba quicksort.py emit-obj -f quicksort_4 -n quicksort_int64_float64_int32_int64 -s "none(int64[:], int64, int64, int64[:], int64[:], float64[:], int32[:], int64[:])" -o quicksort_int64_float64_int32_int64.o

r_61:
		numba quicksort.py emit-obj -f quicksort_4 -n quicksort_int64_float64_int32_int32 -s "none(int64[:], int64, int64, int64[:], int64[:], float64[:], int32[:], int32[:])" -o quicksort_int64_float64_int32_int32.o

r_62:
		numba quicksort.py emit-obj -f quicksort_4 -n quicksort_int64_float64_int32_float64 -s "none(int64[:], int64, int64, int64[:], int64[:], float64[:], int32[:], float64[:])" -o quicksort_int64_float64_int32_float64.o

r_63:
		numba quicksort.py emit-obj -f quicksort_4 -n quicksort_int64_float64_float64_int64 -s "none(int64[:], int64, int64, int64[:], int64[:], float64[:], float64[:], int64[:])" -o quicksort_int64_float64_float64_int64.o

r_64:
		numba quicksort.py emit-obj -f quicksort_4 -n quicksort_int64_float64_float64_int32 -s "none(int64[:], int64, int64, int64[:], int64[:], float64[:], float64[:], int32[:])" -o quicksort_int64_float64_float64_int32.o

r_65:
		numba quicksort.py emit-obj -f quicksort_4 -n quicksort_int64_float64_float64_float64 -s "none(int64[:], int64, int64, int64[:], int64[:], float64[:], float64[:], float64[:])" -o quicksort_int64_float64_float64_float64.o

r_66:
		numba quicksort.py emit-obj -f quicksort_4 -n quicksort_int32_int64_int64_int64 -s "none(int64[:], int64, int64, int64[:], int32[:], int64[:], int64[:], int64[:])" -o quicksort_int32_int64_int64_int64.o

r_67:
		numba quicksort.py emit-obj -f quicksort_4 -n quicksort_int32_int64_int64_int32 -s "none(int64[:], int64, int64, int64[:], int32[:], int64[:], int64[:], int32[:])" -o quicksort_int32_int64_int64_int32.o

r_68:
		numba quicksort.py emit-obj -f quicksort_4 -n quicksort_int32_int64_int64_float64 -s "none(int64[:], int64, int64, int64[:], int32[:], int64[:], int64[:], float64[:])" -o quicksort_int32_int64_int64_float64.o

r_69:
		numba quicksort.py emit-obj -f quicksort_4 -n quicksort_int32_int64_int32_int64 -s "none(int64[:], int64, int64, int64[:], int32[:], int64[:], int32[:], int64[:])" -o quicksort_int32_int64_int32_int64.o

r_70:
		numba quicksort.py emit-obj -f quicksort_4 -n quicksort_int32_int64_int32_int32 -s "none(int64[:], int64, int64, int64[:], int32[:], int64[:], int32[:], int32[:])" -o quicksort_int32_int64_int32_int32.o

r_71:
		numba quicksort.py emit-obj -f quicksort_4 -n quicksort_int32_int64_int32_float64 -s "none(int64[:], int64, int64, int64[:], int32[:], int64[:], int32[:], float64[:])" -o quicksort_int32_int64_int32_float64.o

r_72:
		numba quicksort.py emit-obj -f quicksort_4 -n quicksort_int32_int64_float64_int64 -s "none(int64[:], int64, int64, int64[:], int32[:], int64[:], float64[:], int64[:])" -o quicksort_int32_int64_float64_int64.o

r_73:
		numba quicksort.py emit-obj -f quicksort_4 -n quicksort_int32_int64_float64_int32 -s "none(int64[:], int64, int64, int64[:], int32[:], int64[:], float64[:], int32[:])" -o quicksort_int32_int64_float64_int32.o

r_74:
		numba quicksort.py emit-obj -f quicksort_4 -n quicksort_int32_int64_float64_float64 -s "none(int64[:], int64, int64, int64[:], int32[:], int64[:], float64[:], float64[:])" -o quicksort_int32_int64_float64_float64.o

r_75:
		numba quicksort.py emit-obj -f quicksort_4 -n quicksort_int32_int32_int64_int64 -s "none(int64[:], int64, int64, int64[:], int32[:], int32[:], int64[:], int64[:])" -o quicksort_int32_int32_int64_int64.o

r_76:
		numba quicksort.py emit-obj -f quicksort_4 -n quicksort_int32_int32_int64_int32 -s "none(int64[:], int64, int64, int64[:], int32[:], int32[:], int64[:], int32[:])" -o quicksort_int32_int32_int64_int32.o

r_77:
		numba quicksort.py emit-obj -f quicksort_4 -n quicksort_int32_int32_int64_float64 -s "none(int64[:], int64, int64, int64[:], int32[:], int32[:], int64[:], float64[:])" -o quicksort_int32_int32_int64_float64.o

r_78:
		numba quicksort.py emit-obj -f quicksort_4 -n quicksort_int32_int32_int32_int64 -s "none(int64[:], int64, int64, int64[:], int32[:], int32[:], int32[:], int64[:])" -o quicksort_int32_int32_int32_int64.o

r_79:
		numba quicksort.py emit-obj -f quicksort_4 -n quicksort_int32_int32_int32_int32 -s "none(int64[:], int64, int64, int64[:], int32[:], int32[:], int32[:], int32[:])" -o quicksort_int32_int32_int32_int32.o

r_80:
		numba quicksort.py emit-obj -f quicksort_4 -n quicksort_int32_int32_int32_float64 -s "none(int64[:], int64, int64, int64[:], int32[:], int32[:], int32[:], float64[:])" -o quicksort_int32_int32_int32_float64.o

r_81:
		numba quicksort.py emit-obj -f quicksort_4 -n quicksort_int32_int32_float64_int64 -s "none(int64[:], int64, int64, int64[:], int32[:], int32[:], float64[:], int64[:])" -o quicksort_int32_int32_float64_int64.o

r_82:
		numba quicksort.py emit-obj -f quicksort_4 -n quicksort_int32_int32_float64_int32 -s "none(int64[:], int64, int64, int64[:], int32[:], int32[:], float64[:], int32[:])" -o quicksort_int32_int32_float64_int32.o

r_83:
		numba quicksort.py emit-obj -f quicksort_4 -n quicksort_int32_int32_float64_float64 -s "none(int64[:], int64, int64, int64[:], int32[:], int32[:], float64[:], float64[:])" -o quicksort_int32_int32_float64_float64.o

r_84:
		numba quicksort.py emit-obj -f quicksort_4 -n quicksort_int32_float64_int64_int64 -s "none(int64[:], int64, int64, int64[:], int32[:], float64[:], int64[:], int64[:])" -o quicksort_int32_float64_int64_int64.o

r_85:
		numba quicksort.py emit-obj -f quicksort_4 -n quicksort_int32_float64_int64_int32 -s "none(int64[:], int64, int64, int64[:], int32[:], float64[:], int64[:], int32[:])" -o quicksort_int32_float64_int64_int32.o

r_86:
		numba quicksort.py emit-obj -f quicksort_4 -n quicksort_int32_float64_int64_float64 -s "none(int64[:], int64, int64, int64[:], int32[:], float64[:], int64[:], float64[:])" -o quicksort_int32_float64_int64_float64.o

r_87:
		numba quicksort.py emit-obj -f quicksort_4 -n quicksort_int32_float64_int32_int64 -s "none(int64[:], int64, int64, int64[:], int32[:], float64[:], int32[:], int64[:])" -o quicksort_int32_float64_int32_int64.o

r_88:
		numba quicksort.py emit-obj -f quicksort_4 -n quicksort_int32_float64_int32_int32 -s "none(int64[:], int64, int64, int64[:], int32[:], float64[:], int32[:], int32[:])" -o quicksort_int32_float64_int32_int32.o

r_89:
		numba quicksort.py emit-obj -f quicksort_4 -n quicksort_int32_float64_int32_float64 -s "none(int64[:], int64, int64, int64[:], int32[:], float64[:], int32[:], float64[:])" -o quicksort_int32_float64_int32_float64.o

r_90:
		numba quicksort.py emit-obj -f quicksort_4 -n quicksort_int32_float64_float64_int64 -s "none(int64[:], int64, int64, int64[:], int32[:], float64[:], float64[:], int64[:])" -o quicksort_int32_float64_float64_int64.o

r_91:
		numba quicksort.py emit-obj -f quicksort_4 -n quicksort_int32_float64_float64_int32 -s "none(int64[:], int64, int64, int64[:], int32[:], float64[:], float64[:], int32[:])" -o quicksort_int32_float64_float64_int32.o

r_92:
		numba quicksort.py emit-obj -f quicksort_4 -n quicksort_int32_float64_float64_float64 -s "none(int64[:], int64, int64, int64[:], int32[:], float64[:], float64[:], float64[:])" -o quicksort_int32_float64_float64_float64.o

r_93:
		numba quicksort.py emit-obj -f quicksort_4 -n quicksort_float64_int64_int64_int64 -s "none(int64[:], int64, int64, int64[:], float64[:], int64[:], int64[:], int64[:])" -o quicksort_float64_int64_int64_int64.o

r_94:
		numba quicksort.py emit-obj -f quicksort_4 -n quicksort_float64_int64_int64_int32 -s "none(int64[:], int64, int64, int64[:], float64[:], int64[:], int64[:], int32[:])" -o quicksort_float64_int64_int64_int32.o

r_95:
		numba quicksort.py emit-obj -f quicksort_4 -n quicksort_float64_int64_int64_float64 -s "none(int64[:], int64, int64, int64[:], float64[:], int64[:], int64[:], float64[:])" -o quicksort_float64_int64_int64_float64.o

r_96:
		numba quicksort.py emit-obj -f quicksort_4 -n quicksort_float64_int64_int32_int64 -s "none(int64[:], int64, int64, int64[:], float64[:], int64[:], int32[:], int64[:])" -o quicksort_float64_int64_int32_int64.o

r_97:
		numba quicksort.py emit-obj -f quicksort_4 -n quicksort_float64_int64_int32_int32 -s "none(int64[:], int64, int64, int64[:], float64[:], int64[:], int32[:], int32[:])" -o quicksort_float64_int64_int32_int32.o

r_98:
		numba quicksort.py emit-obj -f quicksort_4 -n quicksort_float64_int64_int32_float64 -s "none(int64[:], int64, int64, int64[:], float64[:], int64[:], int32[:], float64[:])" -o quicksort_float64_int64_int32_float64.o

r_99:
		numba quicksort.py emit-obj -f quicksort_4 -n quicksort_float64_int64_float64_int64 -s "none(int64[:], int64, int64, int64[:], float64[:], int64[:], float64[:], int64[:])" -o quicksort_float64_int64_float64_int64.o

r_100:
		numba quicksort.py emit-obj -f quicksort_4 -n quicksort_float64_int64_float64_int32 -s "none(int64[:], int64, int64, int64[:], float64[:], int64[:], float64[:], int32[:])" -o quicksort_float64_int64_float64_int32.o

r_101:
		numba quicksort.py emit-obj -f quicksort_4 -n quicksort_float64_int64_float64_float64 -s "none(int64[:], int64, int64, int64[:], float64[:], int64[:], float64[:], float64[:])" -o quicksort_float64_int64_float64_float64.o

r_102:
		numba quicksort.py emit-obj -f quicksort_4 -n quicksort_float64_int32_int64_int64 -s "none(int64[:], int64, int64, int64[:], float64[:], int32[:], int64[:], int64[:])" -o quicksort_float64_int32_int64_int64.o

r_103:
		numba quicksort.py emit-obj -f quicksort_4 -n quicksort_float64_int32_int64_int32 -s "none(int64[:], int64, int64, int64[:], float64[:], int32[:], int64[:], int32[:])" -o quicksort_float64_int32_int64_int32.o

r_104:
		numba quicksort.py emit-obj -f quicksort_4 -n quicksort_float64_int32_int64_float64 -s "none(int64[:], int64, int64, int64[:], float64[:], int32[:], int64[:], float64[:])" -o quicksort_float64_int32_int64_float64.o

r_105:
		numba quicksort.py emit-obj -f quicksort_4 -n quicksort_float64_int32_int32_int64 -s "none(int64[:], int64, int64, int64[:], float64[:], int32[:], int32[:], int64[:])" -o quicksort_float64_int32_int32_int64.o

r_106:
		numba quicksort.py emit-obj -f quicksort_4 -n quicksort_float64_int32_int32_int32 -s "none(int64[:], int64, int64, int64[:], float64[:], int32[:], int32[:], int32[:])" -o quicksort_float64_int32_int32_int32.o

r_107:
		numba quicksort.py emit-obj -f quicksort_4 -n quicksort_float64_int32_int32_float64 -s "none(int64[:], int64, int64, int64[:], float64[:], int32[:], int32[:], float64[:])" -o quicksort_float64_int32_int32_float64.o

r_108:
		numba quicksort.py emit-obj -f quicksort_4 -n quicksort_float64_int32_float64_int64 -s "none(int64[:], int64, int64, int64[:], float64[:], int32[:], float64[:], int64[:])" -o quicksort_float64_int32_float64_int64.o

r_109:
		numba quicksort.py emit-obj -f quicksort_4 -n quicksort_float64_int32_float64_int32 -s "none(int64[:], int64, int64, int64[:], float64[:], int32[:], float64[:], int32[:])" -o quicksort_float64_int32_float64_int32.o

r_110:
		numba quicksort.py emit-obj -f quicksort_4 -n quicksort_float64_int32_float64_float64 -s "none(int64[:], int64, int64, int64[:], float64[:], int32[:], float64[:], float64[:])" -o quicksort_float64_int32_float64_float64.o

r_111:
		numba quicksort.py emit-obj -f quicksort_4 -n quicksort_float64_float64_int64_int64 -s "none(int64[:], int64, int64, int64[:], float64[:], float64[:], int64[:], int64[:])" -o quicksort_float64_float64_int64_int64.o

r_112:
		numba quicksort.py emit-obj -f quicksort_4 -n quicksort_float64_float64_int64_int32 -s "none(int64[:], int64, int64, int64[:], float64[:], float64[:], int64[:], int32[:])" -o quicksort_float64_float64_int64_int32.o

r_113:
		numba quicksort.py emit-obj -f quicksort_4 -n quicksort_float64_float64_int64_float64 -s "none(int64[:], int64, int64, int64[:], float64[:], float64[:], int64[:], float64[:])" -o quicksort_float64_float64_int64_float64.o

r_114:
		numba quicksort.py emit-obj -f quicksort_4 -n quicksort_float64_float64_int32_int64 -s "none(int64[:], int64, int64, int64[:], float64[:], float64[:], int32[:], int64[:])" -o quicksort_float64_float64_int32_int64.o

r_115:
		numba quicksort.py emit-obj -f quicksort_4 -n quicksort_float64_float64_int32_int32 -s "none(int64[:], int64, int64, int64[:], float64[:], float64[:], int32[:], int32[:])" -o quicksort_float64_float64_int32_int32.o

r_116:
		numba quicksort.py emit-obj -f quicksort_4 -n quicksort_float64_float64_int32_float64 -s "none(int64[:], int64, int64, int64[:], float64[:], float64[:], int32[:], float64[:])" -o quicksort_float64_float64_int32_float64.o

r_117:
		numba quicksort.py emit-obj -f quicksort_4 -n quicksort_float64_float64_float64_int64 -s "none(int64[:], int64, int64, int64[:], float64[:], float64[:], float64[:], int64[:])" -o quicksort_float64_float64_float64_int64.o

r_118:
		numba quicksort.py emit-obj -f quicksort_4 -n quicksort_float64_float64_float64_int32 -s "none(int64[:], int64, int64, int64[:], float64[:], float64[:], float64[:], int32[:])" -o quicksort_float64_float64_float64_int32.o

r_119:
		numba quicksort.py emit-obj -f quicksort_4 -n quicksort_float64_float64_float64_float64 -s "none(int64[:], int64, int64, int64[:], float64[:], float64[:], float64[:], float64[:])" -o quicksort_float64_float64_float64_float64.o