import pandas as pd
import numpy as np
import argparse
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder

EPS = np.float64(1e-12)

columnList = [
    'Hospital Service Area', 
 'Hospital County', 
 'Operating Certificate Number', 
 'Permanent Facility Id', 
 'Facility Name', 
 'Age Group', 
 'Zip Code - 3 digits', 
 'Total Costs',   
 'Race', 
 'Ethnicity', 
 'Length of Stay', 
 'Type of Admission',
 'Patient Disposition', 
 'CCSR Diagnosis Code', 
 'CCSR Procedure Code', 
 'APR DRG Code', 
 'APR MDC Code', 
 'APR Severity of Illness Description',
 'APR Risk of Mortality', 
 'APR Medical Surgical Description', 
 'Payment Typology 1',  
 'Payment Typology 2', 
 'Payment Typology 3', 
 'Birth Weight',  
 'Emergency Department Indicator' 
 ]
 
targetColumnList = [
     'Operating Certificate Number',
 'Length of Stay',
 'Birth Weight',
     'Total Costs'
 ]

oneHotColumnList= [
'Hospital Service Area', 
 'Hospital County',   
 'Permanent Facility Id', 
 'Facility Name',  
 'Age Group', 
 'Zip Code - 3 digits',   
 'Race', 
 'Ethnicity', 
 'Type of Admission',
 'Patient Disposition', 
 'CCSR Diagnosis Code', 
 'CCSR Procedure Code', 
 'APR DRG Code', 
 'APR MDC Code', 
 'APR Severity of Illness Description',
 'APR Risk of Mortality',
 'APR Medical Surgical Description', 
 'Payment Typology 1', 
 'Payment Typology 2', 
 'Payment Typology 3', 
 'Emergency Department Indicator' ]

selectedColumns = ['CCSR Diagnosis Code_458', 'CCSR Procedure Code_221', 'CCSR Diagnosis Code_457', 
                   'Zip Code - 3 digits_47', 'CCSR Procedure Code_96', 'CCSR Diagnosis Code_455', 
                   'CCSR Procedure Code_100', 'CCSR Procedure Code_206', 'CCSR Procedure Code_107', 
                   'CCSR Diagnosis Code_468', 'Zip Code - 3 digits_5', 'CCSR Procedure Code_117', 
                   'CCSR Procedure Code_115', 'Zip Code - 3 digits_2', 'CCSR Diagnosis Code_479', 
                   'CCSR Diagnosis Code_478', 'CCSR Procedure Code_114', 'CCSR Diagnosis Code_477', 
                   'CCSR Procedure Code_189', 'CCSR Procedure Code_112', 'CCSR Procedure Code_193', 
                   'CCSR Diagnosis Code_473', 'CCSR Procedure Code_110', 'CCSR Procedure Code_196', 
                   'CCSR Procedure Code_197', 'CCSR Procedure Code_198', 'CCSR Procedure Code_199', 
                   'CCSR Procedure Code_200', 'CCSR Diagnosis Code_471', 'CCSR Procedure Code_109', 
                   'Zip Code - 3 digits_43', 'CCSR Diagnosis Code_454', 'CCSR Procedure Code_319', 
                   'CCSR Procedure Code_135', 'APR MDC Code_4', 'CCSR Diagnosis Code_359',
                   'CCSR Procedure Code_19', 'CCSR Procedure Code_18', 'CCSR Procedure Code_16', 
                   'Permanent Facility Id_779.0', 'Permanent Facility Id_798.0', 'CCSR Diagnosis Code_354',
                   'CCSR Diagnosis Code_353', 'CCSR Diagnosis Code_352', 'Permanent Facility Id_818.0', 
                   'CCSR Procedure Code_15', 'Permanent Facility Id_848.0', 'CCSR Diagnosis Code_350', 
                   'Permanent Facility Id_870.0', 'CCSR Procedure Code_12', 'Permanent Facility Id_891.0', 
                   'CCSR Diagnosis Code_346', 'Permanent Facility Id_727.0', 'CCSR Diagnosis Code_361', 
                   'CCSR Diagnosis Code_362', 'CCSR Diagnosis Code_363', 'APR MDC Code_18', 
                   'CCSR Procedure Code_33', 'Permanent Facility Id_552.0', 'CCSR Procedure Code_32', 
                   'CCSR Diagnosis Code_372', 'APR MDC Code_20', 'Permanent Facility Id_583.0', 
                   'APR MDC Code_22', 'CCSR Procedure Code_10', 'APR MDC Code_25', 'CCSR Procedure Code_25',
                   'Permanent Facility Id_630.0', 'Permanent Facility Id_635.0', 'CCSR Procedure Code_58', 
                   'CCSR Procedure Code_24', 'CCSR Procedure Code_23', 'Permanent Facility Id_678.0', 
                   'Permanent Facility Id_694.0', 'CCSR Diagnosis Code_369', 'CCSR Procedure Code_8', 
                   'Permanent Facility Id_924.0', 'Permanent Facility Id_938.0', 'Permanent Facility Id_1153.0',
                   'CCSR Diagnosis Code_327', 'Type of Admission_5', 'Type of Admission_4', 
                   'Permanent Facility Id_1164.0', 'Permanent Facility Id_1165.0', 'CCSR Diagnosis Code_326', 
                   'Permanent Facility Id_1169.0', 'Permanent Facility Id_1139.0', 'Type of Admission_3',
                   'Permanent Facility Id_1176.0', 'Permanent Facility Id_1178.0', 'Type of Admission_2', 
                   'Permanent Facility Id_1286.0', 'Permanent Facility Id_1288.0', 'Patient Disposition_18', 
                   'Total Costs', 'Permanent Facility Id_1301.0', 'Permanent Facility Id_1175.0', 
                   'Permanent Facility Id_550.0', 'Permanent Facility Id_1138.0', 'Permanent Facility Id_1124.0',
                   'Permanent Facility Id_943.0', 'Permanent Facility Id_968.0', 'Permanent Facility Id_971.0',
                   'CCSR Diagnosis Code_341', 'CCSR Diagnosis Code_340', 'CCSR Procedure Code_7', 
                   'CCSR Diagnosis Code_337', 'Permanent Facility Id_1028.0', 'CCSR Diagnosis Code_328', 
                   'Patient Disposition_11', 'CCSR Diagnosis Code_334', 'Permanent Facility Id_1061.0', 
                   'Patient Disposition_14', 'CCSR Diagnosis Code_332', 'Race_2', 'Permanent Facility Id_1099.0',
                   'CCSR Diagnosis Code_330', 'Patient Disposition_16', 'CCSR Procedure Code_4', 
                   'Permanent Facility Id_528.0', 'Permanent Facility Id_636.0', 'Permanent Facility Id_98.0', 
                   'Permanent Facility Id_324.0', 'CCSR Diagnosis Code_388', 'Permanent Facility Id_330.0', 
                   'APR MDC Code_11', 'CCSR Procedure Code_48', 'Permanent Facility Id_128.0', 
                   'CCSR Procedure Code_39', 'Permanent Facility Id_339.0', 'CCSR Procedure Code_49', 
                   'Permanent Facility Id_103.0', 'CCSR Diagnosis Code_386', 'Permanent Facility Id_208.0', 
                   'APR MDC Code_10', 'Permanent Facility Id_367.0', 'Permanent Facility Id_116.0', 
                   'Permanent Facility Id_377.0', 'Permanent Facility Id_292.0', 'CCSR Procedure Code_41', 
                   'CCSR Procedure Code_43', 'CCSR Diagnosis Code_395', 'Permanent Facility Id_181.0', 
                   'Permanent Facility Id_180.0', 'CCSR Procedure Code_44', 'CCSR Procedure Code_45', 
                   'CCSR Procedure Code_40', 'CCSR Diagnosis Code_392', 'APR MDC Code_12', 
                   'CCSR Diagnosis Code_377', 'CCSR Diagnosis Code_391', 'Permanent Facility Id_245.0', 
                   'Permanent Facility Id_146.0', 'CCSR Diagnosis Code_390', 'CCSR Diagnosis Code_397', 
                   'CCSR Diagnosis Code_401', 'Permanent Facility Id_207.0', 'CCSR Diagnosis Code_382', 
                   'APR MDC Code_14', 'Ethnicity_3', 'Ethnicity_4', 'Emergency Department Indicator_2', 
                   'Permanent Facility Id_471.0', 'Permanent Facility Id_484.0', 'Permanent Facility Id_412.0',
                   'Permanent Facility Id_39.0', 'Ethnicity_2', 'Permanent Facility Id_42.0',
                   'Permanent Facility Id_401.0', 'CCSR Diagnosis Code_402', 'Permanent Facility Id_397.0', 
                   'CCSR Diagnosis Code_381', 'Permanent Facility Id_58.0', 'Permanent Facility Id_66.0', 
                   'Permanent Facility Id_383.0', 'Permanent Facility Id_518.0', 'Permanent Facility Id_409.0', 
                   'Permanent Facility Id_490.0', 'Patient Disposition_17', 'Patient Disposition_10', 'APR DRG Code_811', 'APR DRG Code_844', 'APR DRG Code_816', 'CCSR Diagnosis Code_33', 'APR MDC Code_6', 'CCSR Diagnosis Code_85', 'CCSR Diagnosis Code_86', 'APR MDC Code_5', 'Zip Code - 3 digits_39', 'CCSR Diagnosis Code_43', 'Patient Disposition_15', 'CCSR Diagnosis Code_42', 'CCSR Diagnosis Code_79', 'CCSR Diagnosis Code_77', 'CCSR Diagnosis Code_31', 'Zip Code - 3 digits_37', 'CCSR Diagnosis Code_29', 'APR MDC Code_9', 'Patient Disposition_12', 'APR MDC Code_2', 'APR MDC Code_7', 'CCSR Diagnosis Code_27', 'Patient Disposition_13', 'CCSR Diagnosis Code_84', 'CCSR Diagnosis Code_76', 'Zip Code - 3 digits_34', 'APR MDC Code_8', 'Payment Typology 1_3', 'Patient Disposition_8', 'APR DRG Code_850', 'CCSR Diagnosis Code_4', 'CCSR Diagnosis Code_60', 'CCSR Diagnosis Code_18', 'CCSR Diagnosis Code_6', 'CCSR Diagnosis Code_58', 'Zip Code - 3 digits_48', 'CCSR Diagnosis Code_57', 'APR MDC Code_15', 'CCSR Diagnosis Code_55', 'APR MDC Code_23', 'CCSR Diagnosis Code_17', 'Zip Code - 3 digits_49', 'APR MDC Code_21', 'Payment Typology 1_4', 'Zip Code - 3 digits_50', 'CCSR Diagnosis Code_49', 'APR MDC Code_19', 'CCSR Diagnosis Code_53', 'Payment Typology 1_2', 'CCSR Diagnosis Code_16', 'APR MDC Code_16', 'CCSR Diagnosis Code_14', 'APR MDC Code_17', 'CCSR Diagnosis Code_12', 'CCSR Diagnosis Code_13', 'Patient Disposition_2', 'CCSR Diagnosis Code_74', 'Zip Code - 3 digits_46', 'CCSR Diagnosis Code_62', 'CCSR Diagnosis Code_72', 'Zip Code - 3 digits_40', 'CCSR Diagnosis Code_24', 'CCSR Diagnosis Code_69', 'APR DRG Code_860', 'APR DRG Code_862', 'Patient Disposition_9', 'Payment Typology 1_5', 'Zip Code - 3 digits_41', 'CCSR Diagnosis Code_67', 'Zip Code - 3 digits_42', 'CCSR Diagnosis Code_23', 'Patient Disposition_7', 'CCSR Diagnosis Code_46', 'Patient Disposition_6', 'CCSR Diagnosis Code_47', 'APR MDC Code_13', 'Patient Disposition_5', 'APR DRG Code_950', 'CCSR Diagnosis Code_48', 'CCSR Diagnosis Code_64', 'Patient Disposition_4', 'CCSR Diagnosis Code_21', 'APR DRG Code_952', 'Patient Disposition_3', 'CCSR Diagnosis Code_20', 'Payment Typology 1_9', 'CCSR Diagnosis Code_321', 'Zip Code - 3 digits_31', 'CCSR Diagnosis Code_244', 
                   'CCSR Diagnosis Code_387', 'CCSR Diagnosis Code_246', 'CCSR Diagnosis Code_385', 'CCSR Diagnosis Code_384', 'CCSR Diagnosis Code_383', 'CCSR Diagnosis Code_249', 'CCSR Diagnosis Code_380', 'CCSR Diagnosis Code_378', 'CCSR Diagnosis Code_252', 'CCSR Diagnosis Code_254', 'CCSR Diagnosis Code_376', 'CCSR Diagnosis Code_255', 'CCSR Diagnosis Code_256', 'CCSR Diagnosis Code_375', 'CCSR Diagnosis Code_374', 'CCSR Diagnosis Code_373', 'CCSR Diagnosis Code_259', 'CCSR Diagnosis Code_262', 'CCSR Diagnosis Code_243', 'CCSR Diagnosis Code_389', 'CCSR Diagnosis Code_242', 'CCSR Diagnosis Code_240', 'CCSR Diagnosis Code_207', 'CCSR Diagnosis Code_209', 'CCSR Diagnosis Code_404', 'CCSR Diagnosis Code_211', 'CCSR Diagnosis Code_403', 'CCSR Diagnosis Code_217', 'CCSR Diagnosis Code_218', 'CCSR Diagnosis Code_223', 'CCSR Diagnosis Code_224', 'CCSR Diagnosis Code_371', 'CCSR Diagnosis Code_226', 'CCSR Diagnosis Code_232', 'CCSR Diagnosis Code_233', 'CCSR Diagnosis Code_400', 'CCSR Diagnosis Code_234', 'CCSR Diagnosis Code_399', 'CCSR Diagnosis Code_396', 'CCSR Diagnosis Code_236', 'CCSR Diagnosis Code_394', 'CCSR Diagnosis Code_393', 'CCSR Diagnosis Code_230', 'CCSR Diagnosis Code_263', 'CCSR Diagnosis Code_370', 'CCSR Diagnosis Code_368', 'CCSR Diagnosis Code_338', 'CCSR Diagnosis Code_298', 'CCSR Diagnosis Code_301', 'CCSR Diagnosis Code_302', 'CCSR Diagnosis Code_336', 'CCSR Diagnosis Code_335', 'CCSR Diagnosis Code_304', 'CCSR Diagnosis Code_333', 'CCSR Diagnosis Code_305', 'CCSR Diagnosis Code_342', 'CCSR Diagnosis Code_331', 'CCSR Diagnosis Code_309', 'CCSR Diagnosis Code_310', 'CCSR Diagnosis Code_311', 'CCSR Diagnosis Code_325', 'CCSR Diagnosis Code_324', 'CCSR Diagnosis Code_314', 'CCSR Diagnosis Code_315', 'CCSR Diagnosis Code_323', 'CCSR Diagnosis Code_322', 'CCSR Diagnosis Code_329', 'CCSR Diagnosis Code_406', 'CCSR Diagnosis Code_294', 'CCSR Diagnosis Code_292', 'CCSR Diagnosis Code_267', 'CCSR Diagnosis Code_367', 'CCSR Diagnosis Code_366', 'CCSR Diagnosis Code_364', 'CCSR Diagnosis Code_273', 'CCSR Diagnosis Code_360', 'CCSR Diagnosis Code_277', 'CCSR Diagnosis Code_278', 'CCSR Diagnosis Code_358', 'CCSR Diagnosis Code_293', 'CCSR Diagnosis Code_279', 'CCSR Diagnosis Code_355', 'CCSR Diagnosis Code_285', 'CCSR Diagnosis Code_351', 'CCSR Diagnosis Code_288', 'CCSR Diagnosis Code_349', 'CCSR Diagnosis Code_347', 'CCSR Diagnosis Code_290', 'CCSR Diagnosis Code_345', 'CCSR Diagnosis Code_343', 'CCSR Diagnosis Code_280', 'CCSR Diagnosis Code_206', 'CCSR Diagnosis Code_205', 'CCSR Diagnosis Code_407', 'Zip Code - 3 digits_4', 'Zip Code - 3 digits_3', 'CCSR Diagnosis Code_122', 'CCSR Diagnosis Code_124', 'CCSR Diagnosis Code_126', 'CCSR Diagnosis Code_476', 'CCSR Diagnosis Code_475', 'CCSR Diagnosis Code_474', 'CCSR Diagnosis Code_472', 'CCSR Diagnosis Code_121', 'CCSR Diagnosis Code_470', 'CCSR Diagnosis Code_467', 'CCSR Diagnosis Code_135', 'CCSR Diagnosis Code_136', 'CCSR Diagnosis Code_466', 'CCSR Diagnosis Code_138', 'CCSR Diagnosis Code_139', 'CCSR Diagnosis Code_462', 'CCSR Diagnosis Code_461', 'CCSR Diagnosis Code_141', 'CCSR Diagnosis Code_469', 'CCSR Diagnosis Code_460', 'CCSR Diagnosis Code_120', 'CCSR Diagnosis Code_118', 'CCSR Diagnosis Code_89', 'Zip Code - 3 digits_29', 'CCSR Diagnosis Code_91', 'Zip Code - 3 digits_28', 'CCSR Diagnosis Code_92', 'CCSR Diagnosis Code_93', 'CCSR Diagnosis Code_95', 'CCSR Diagnosis Code_97', 'CCSR Diagnosis Code_98', 'CCSR Diagnosis Code_119', 'Zip Code - 3 digits_24', 'Zip Code - 3 digits_19', 'Zip Code - 3 digits_18', 'Zip Code - 3 digits_17', 'CCSR Diagnosis Code_106', 'CCSR Diagnosis Code_107', 'CCSR Diagnosis Code_108', 'Zip Code - 3 digits_13', 'CCSR Diagnosis Code_111', 'CCSR Diagnosis Code_112', 'Zip Code - 3 digits_23', 'Zip Code - 3 digits_33', 'CCSR Diagnosis Code_144', 'CCSR Diagnosis Code_147', 'CCSR Diagnosis Code_180', 'CCSR Diagnosis Code_183', 'CCSR Diagnosis Code_186', 'CCSR Diagnosis Code_187', 'CCSR Diagnosis Code_189', 'CCSR Diagnosis Code_430', 'CCSR Diagnosis Code_429', 'CCSR Diagnosis Code_428', 'CCSR Diagnosis Code_426', 'CCSR Diagnosis Code_179', 'CCSR Diagnosis Code_191', 'CCSR Diagnosis Code_192', 'CCSR Diagnosis Code_424', 'CCSR Diagnosis Code_193', 'CCSR Diagnosis Code_422', 'CCSR Diagnosis Code_421', 'CCSR Diagnosis Code_410', 'CCSR Diagnosis Code_196', 'CCSR Diagnosis Code_198', 'CCSR Diagnosis Code_408', 'CCSR Diagnosis Code_425', 'CCSR Diagnosis Code_459', 'CCSR Diagnosis Code_178', 'CCSR Diagnosis Code_173', 'CCSR Diagnosis Code_456', 'CCSR Diagnosis Code_149', 'CCSR Diagnosis Code_150', 'CCSR Diagnosis Code_152', 'CCSR Diagnosis Code_153', 'CCSR Diagnosis Code_154', 'CCSR Diagnosis Code_453', 'CCSR Diagnosis Code_452', 'CCSR Diagnosis Code_451', 'CCSR Diagnosis Code_434', 'CCSR Diagnosis Code_155', 'CCSR Diagnosis Code_160', 'CCSR Diagnosis Code_161', 'CCSR Diagnosis Code_162', 'CCSR Diagnosis Code_449', 'CCSR Diagnosis Code_164', 'CCSR Diagnosis Code_446', 'CCSR Diagnosis Code_444', 'CCSR Diagnosis Code_169', 'CCSR Diagnosis Code_170', 'CCSR Diagnosis Code_159', 'Operating Certificate Number', 'APR DRG Code_53', 'APR DRG Code_774', 'Permanent Facility Id_192.0', 'Permanent Facility Id_210.0', 'Permanent Facility Id_213.0', 'Permanent Facility Id_216.0', 'Permanent Facility Id_218.0', 'Permanent Facility Id_267.0', 'Permanent Facility Id_280.0', 'Permanent Facility Id_325.0', 'Permanent Facility Id_340.0', 'Permanent Facility Id_362.0', 'Permanent Facility Id_379.0', 'Permanent Facility Id_393.0', 'Permanent Facility Id_411.0', 'Permanent Facility Id_413.0', 'Permanent Facility Id_482.0', 'Permanent Facility Id_511.0', 'Permanent Facility Id_513.0', 'Permanent Facility Id_527.0', 'Permanent Facility Id_541.0', 'Permanent Facility Id_551.0', 'Permanent Facility Id_563.0', 'Permanent Facility Id_565.0', 'Permanent Facility Id_574.0', 'Permanent Facility Id_589.0', 'Permanent Facility Id_598.0', 'Permanent Facility Id_599.0', 'Permanent Facility Id_628.0', 'Permanent Facility Id_671.0', 'Permanent Facility Id_676.0', 'Permanent Facility Id_174.0', 'Permanent Facility Id_170.0', 'Permanent Facility Id_158.0', 'Permanent Facility Id_135.0', 'CCSR Procedure Code_261', 'CCSR Procedure Code_262', 'CCSR Procedure Code_265', 'CCSR Procedure Code_266', 'CCSR Procedure Code_268', 'CCSR Procedure Code_269', 'CCSR Procedure Code_276', 'CCSR Procedure Code_278', 'CCSR Procedure Code_282', 'CCSR Procedure Code_283', 'CCSR Procedure Code_284', 'CCSR Procedure Code_285', 'CCSR Procedure Code_288', 'CCSR Procedure Code_290', 'Permanent Facility Id_699.0', 'CCSR Procedure Code_293', 'CCSR Procedure Code_296', 'CCSR Procedure Code_297', 'CCSR Procedure Code_300', 'CCSR Procedure Code_302', 'CCSR Procedure Code_304', 'CCSR Procedure Code_305', 'CCSR Procedure Code_314', 'CCSR Procedure Code_316', 'CCSR Procedure Code_317', 'CCSR Procedure Code_318', 'CCSR Procedure Code_320', 'CCSR Procedure Code_321', 'Permanent Facility Id_43.0', 'Permanent Facility Id_85.0', 'CCSR Procedure Code_294', 'Permanent Facility Id_704.0', 'Permanent Facility Id_708.0', 'Permanent Facility Id_718.0', 'Permanent Facility Id_1304.0', 'Permanent Facility Id_1306.0', 'Permanent Facility Id_1309.0', 'Permanent Facility Id_1315.0', 'Permanent Facility Id_1318.0', 'Permanent Facility Id_1320.0', 'Permanent Facility Id_1324.0', 'Permanent Facility Id_1437.0', 'Permanent Facility Id_1453.0', 'Permanent Facility Id_1454.0', 'Permanent Facility Id_1460.0', 'Permanent Facility Id_1463.0', 'Permanent Facility Id_1466.0', 'Permanent Facility Id_1626.0', 'Permanent Facility Id_1294.0', 'Permanent Facility Id_1633.0', 'Permanent Facility Id_1639.0', 'Permanent Facility Id_1737.0', 'Permanent Facility Id_3067.0', 'Permanent Facility Id_3376.0', 'Permanent Facility Id_3975.0', 'Permanent Facility Id_9059.0', 'Permanent Facility Id_10223.0', 'Facility Name_4', 'Facility Name_5', 'Facility Name_11', 'Facility Name_12', 'Facility Name_13', 'Facility Name_15', 'Facility Name_16', 'Permanent Facility Id_1638.0', 'CCSR Procedure Code_259', 'Permanent Facility Id_1293.0', 'Permanent Facility Id_1172.0', 'Permanent Facility Id_739.0', 'Permanent Facility Id_746.0', 'Permanent Facility Id_752.0', 'Permanent Facility Id_776.0', 'Permanent Facility Id_804.0', 'Permanent Facility Id_812.0', 'Permanent Facility Id_815.0', 'Permanent Facility Id_829.0', 'Permanent Facility Id_866.0', 'Permanent Facility Id_885.0', 'Permanent Facility Id_889.0', 'Permanent Facility Id_895.0', 'Permanent Facility Id_896.0', 'Permanent Facility Id_913.0', 'Permanent Facility Id_1186.0', 'Permanent Facility Id_925.0', 'Permanent Facility Id_989.0', 'Permanent Facility Id_990.0', 'Permanent Facility Id_1005.0', 'Permanent Facility Id_1039.0', 'Permanent Facility Id_1045.0', 'Permanent Facility Id_1047.0', 'Permanent Facility Id_1072.0', 'Permanent Facility Id_1097.0', 'Permanent Facility Id_1098.0', 'Permanent Facility Id_1117.0', 'Permanent Facility Id_1122.0', 'Permanent Facility Id_1129.0', 'Permanent Facility Id_1158.0', 'Permanent Facility Id_1168.0', 'Permanent Facility Id_977.0', 'CCSR Procedure Code_253', 'CCSR Procedure Code_248', 'CCSR Procedure Code_245', 'CCSR Procedure Code_54', 'CCSR Procedure Code_55', 'CCSR Procedure Code_56', 'CCSR Procedure Code_57', 'CCSR Procedure Code_59', 'CCSR Procedure Code_63', 'CCSR Procedure Code_66', 'CCSR Procedure Code_68', 'CCSR Procedure Code_69', 'CCSR Procedure Code_73', 'CCSR Procedure Code_76', 'CCSR Procedure Code_82', 'CCSR Procedure Code_84', 'CCSR Procedure Code_88', 'CCSR Procedure Code_53', 'CCSR Procedure Code_90', 'CCSR Procedure Code_95', 'CCSR Procedure Code_97', 'CCSR Procedure Code_101', 'CCSR Procedure Code_102', 'CCSR Procedure Code_104', 'CCSR Procedure Code_108', 'CCSR Procedure Code_111', 'CCSR Procedure Code_113', 'CCSR Procedure Code_116', 'CCSR Procedure Code_119', 'CCSR Procedure Code_123', 'CCSR Procedure Code_126', 'CCSR Procedure Code_128', 'CCSR Procedure Code_129', 'CCSR Procedure Code_92', 'CCSR Procedure Code_133', 'CCSR Procedure Code_51', 'CCSR Procedure Code_47', 'Length of Stay', 'Birth Weight', 'Type of Admission_6', 'Race_3', 'Race_4', 'CCSR Procedure Code_2', 'CCSR Procedure Code_3', 'CCSR Procedure Code_5', 'CCSR Procedure Code_6', 'CCSR Procedure Code_9', 'CCSR Procedure Code_11', 'CCSR Procedure Code_13', 'CCSR Procedure Code_14', 'CCSR Procedure Code_17', 'CCSR Procedure Code_50', 'CCSR Procedure Code_20', 'CCSR Procedure Code_22', 'CCSR Procedure Code_26', 'CCSR Procedure Code_27', 'CCSR Procedure Code_28', 'CCSR Procedure Code_29', 'CCSR Procedure Code_30', 'CCSR Procedure Code_31', 'CCSR Procedure Code_34', 'CCSR Procedure Code_35', 'CCSR Procedure Code_36', 'CCSR Procedure Code_37', 'CCSR Procedure Code_38', 'CCSR Procedure Code_42', 'CCSR Procedure Code_46', 'CCSR Procedure Code_21', 'Facility Name_19', 'CCSR Procedure Code_136', 'CCSR Procedure Code_138', 'CCSR Procedure Code_192', 'CCSR Procedure Code_194', 'CCSR Procedure Code_195', 'CCSR Procedure Code_201', 'CCSR Procedure Code_202', 'CCSR Procedure Code_203', 'CCSR Procedure Code_204', 'CCSR Procedure Code_205', 'CCSR Procedure Code_208', 'CCSR Procedure Code_210', 'CCSR Procedure Code_212', 'CCSR Procedure Code_213', 'CCSR Procedure Code_214', 'CCSR Procedure Code_215', 'CCSR Procedure Code_191', 'CCSR Procedure Code_216', 'CCSR Procedure Code_220', 'CCSR Procedure Code_222', 'CCSR Procedure Code_223', 'CCSR Procedure Code_224', 'CCSR Procedure Code_226', 'CCSR Procedure Code_229', 'CCSR Procedure Code_231', 'CCSR Procedure Code_232', 'CCSR Procedure Code_235', 'CCSR Procedure Code_237', 'CCSR Procedure Code_240', 'CCSR Procedure Code_241', 'CCSR Procedure Code_243', 'CCSR Procedure Code_244', 'CCSR Procedure Code_219', 'CCSR Procedure Code_137', 'CCSR Procedure Code_190', 'CCSR Procedure Code_187', 'CCSR Procedure Code_139', 'CCSR Procedure Code_144', 'CCSR Procedure Code_145', 'CCSR Procedure Code_148', 'CCSR Procedure Code_150', 'CCSR Procedure Code_152', 'CCSR Procedure Code_154', 'CCSR Procedure Code_156', 'CCSR Procedure Code_157', 'CCSR Procedure Code_159', 'CCSR Procedure Code_160', 'CCSR Procedure Code_161', 'CCSR Procedure Code_162', 'CCSR Procedure Code_163', 'CCSR Procedure Code_188', 'CCSR Procedure Code_166', 'CCSR Procedure Code_168', 'CCSR Procedure Code_169', 'CCSR Procedure Code_173', 'CCSR Procedure Code_174', 'CCSR Procedure Code_175', 'CCSR Procedure Code_176', 'CCSR Procedure Code_178', 'CCSR Procedure Code_179', 'CCSR Procedure Code_180', 'CCSR Procedure Code_181', 'CCSR Procedure Code_183', 'CCSR Procedure Code_184', 'CCSR Procedure Code_185', 'CCSR Procedure Code_186', 'CCSR Procedure Code_167', 'Facility Name_23', 'Facility Name_25', 'Facility Name_28', 'APR DRG Code_243', 'APR DRG Code_245', 'APR DRG Code_246', 'APR DRG Code_253', 'APR DRG Code_261', 'APR DRG Code_263', 'APR DRG Code_279', 'APR DRG Code_280', 'APR DRG Code_281', 'APR DRG Code_282', 'APR DRG Code_283', 'APR DRG Code_284', 'APR DRG Code_305', 'APR DRG Code_309', 'APR DRG Code_241', 'APR DRG Code_313', 'APR DRG Code_320', 'APR DRG Code_321', 'APR DRG Code_323', 'APR DRG Code_326', 'APR DRG Code_340', 'APR DRG Code_341', 'APR DRG Code_344', 'APR DRG Code_346', 'APR DRG Code_347', 'APR DRG Code_361', 'APR DRG Code_364', 'APR DRG Code_382', 'APR DRG Code_383', 'APR DRG Code_385', 'APR DRG Code_317', 'APR DRG Code_401', 'APR DRG Code_240', 'APR DRG Code_231', 'APR DRG Code_110', 'APR DRG Code_114', 'APR DRG Code_115', 'APR DRG Code_121', 'APR DRG Code_131', 'APR DRG Code_132', 'APR DRG Code_133', 'APR DRG Code_137', 'APR DRG Code_139', 'APR DRG Code_143', 'APR DRG Code_160', 'APR DRG Code_161', 'APR DRG Code_163', 'APR DRG Code_169', 'APR DRG Code_233', 'APR DRG Code_176', 'APR DRG Code_181', 'APR DRG Code_182', 'APR DRG Code_183', 'APR DRG Code_190', 'APR DRG Code_191', 'APR DRG Code_192', 'APR DRG Code_198', 'APR DRG Code_200', 'APR DRG Code_203', 'APR DRG Code_222', 'APR DRG Code_223', 'APR DRG Code_226', 'APR DRG Code_227', 'APR DRG Code_230', 'APR DRG Code_179', 'APR DRG Code_95', 'APR DRG Code_422', 'APR DRG Code_425', 'APR DRG Code_612', 'APR DRG Code_614', 'APR DRG Code_626', 'APR DRG Code_631', 'APR DRG Code_633', 'APR DRG Code_636', 'APR DRG Code_639', 'APR DRG Code_640', 'APR DRG Code_650', 'APR DRG Code_651', 'APR DRG Code_660', 'APR DRG Code_661', 'APR DRG Code_680', 'APR DRG Code_681', 'APR DRG Code_611', 'APR DRG Code_695', 'APR DRG Code_710', 'APR DRG Code_711', 'APR DRG Code_720', 'APR DRG Code_721', 'APR DRG Code_722', 'APR DRG Code_723', 'APR DRG Code_740', 'APR DRG Code_752', 'APR DRG Code_753', 'APR DRG Code_756', 'APR DRG Code_759', 'APR DRG Code_760', 'APR DRG Code_772', 'APR DRG Code_773', 'APR DRG Code_696', 'APR DRG Code_424', 'APR DRG Code_609', 'APR DRG Code_607', 'APR DRG Code_441', 'APR DRG Code_442', 'APR DRG Code_445', 'APR DRG Code_446', 'APR DRG Code_447', 'APR DRG Code_462', 'APR DRG Code_463', 'APR DRG Code_465', 'APR DRG Code_466', 'APR DRG Code_470', 'APR DRG Code_482', 'APR DRG Code_484', 'APR DRG Code_510', 'APR DRG Code_512', 'APR DRG Code_608', 'APR DRG Code_514', 'APR DRG Code_519', 'APR DRG Code_530', 'APR DRG Code_532', 'APR DRG Code_539', 'APR DRG Code_541', 'APR DRG Code_547', 'APR DRG Code_548', 'APR DRG Code_566', 'APR DRG Code_581', 'APR DRG Code_588', 'APR DRG Code_589', 'APR DRG Code_591', 'APR DRG Code_602', 'APR DRG Code_603', 'APR DRG Code_518', 'APR DRG Code_810', 'APR DRG Code_92', 'APR DRG Code_89', 'Facility Name_110', 'Facility Name_112', 'Facility Name_113', 'Facility Name_115', 'Facility Name_117', 'Facility Name_122', 'Facility Name_123', 'Facility Name_126', 'Facility Name_127', 'Facility Name_131', 'Facility Name_134', 'Facility Name_136', 'Facility Name_137', 'Facility Name_138', 'Facility Name_108', 'Facility Name_141', 'Facility Name_146', 'Facility Name_151', 'Facility Name_152', 'Facility Name_155', 'Facility Name_156', 'Facility Name_157', 'Facility Name_159', 'Facility Name_161', 'Facility Name_162', 'Facility Name_166', 'Facility Name_167', 'Facility Name_169', 'Facility Name_172', 'Facility Name_173', 'Facility Name_143', 'Facility Name_174', 'Facility Name_102', 'Facility Name_92', 'Facility Name_34', 'Facility Name_35', 'Facility Name_36', 'Facility Name_38', 'Facility Name_40', 'Facility Name_42', 'Facility Name_45', 'Facility Name_46', 'Facility Name_47', 'Facility Name_48', 'Facility Name_49', 'Facility Name_51', 'Facility Name_53', 'Facility Name_54', 'Facility Name_101', 'Facility Name_57', 'Facility Name_62', 'Facility Name_65', 'Facility Name_67', 'Facility Name_69', 'Facility Name_71', 'Facility Name_74', 'Facility Name_76', 'Facility Name_78', 'Facility Name_80', 'Facility Name_82', 'Facility Name_83', 'Facility Name_84', 'Facility Name_88', 'Facility Name_89', 'Facility Name_61', 'APR DRG Code_91', 'Facility Name_175', 'Facility Name_178', 'Hospital Service Area_3', 'Hospital Service Area_4', 'Hospital Service Area_5', 'Hospital Service Area_7', 'Hospital Service Area_8', 'Payment Typology 3_8.0', 'APR Severity of Illness Description_2', 'APR Severity of Illness Description_4', 'Age Group_3', 'Age Group_4', 'Age Group_5', 'APR Risk of Mortality_2', 'APR Risk of Mortality_3', 'APR DRG Code_2', 'Hospital Service Area_2', 'APR DRG Code_5', 'APR DRG Code_9', 'APR DRG Code_20', 'APR DRG Code_26', 'APR DRG Code_30', 'APR DRG Code_40', 'APR DRG Code_42', 'APR DRG Code_43', 'APR DRG Code_44', 'APR DRG Code_46', 'APR DRG Code_47', 'APR DRG Code_48', 'APR DRG Code_55', 'APR DRG Code_58', 'APR DRG Code_73', 'APR DRG Code_8', 'Facility Name_177', 'Hospital County_56', 'Hospital County_52', 'Facility Name_181', 'Facility Name_185', 'Facility Name_188', 'Facility Name_193', 'Facility Name_199', 'Payment Typology 2_2.0', 'Payment Typology 2_7.0', 'Hospital County_3', 'Hospital County_4', 'Hospital County_7', 'Hospital County_9', 'Hospital County_10', 'Hospital County_11', 'Hospital County_12', 'Hospital County_55', 'Hospital County_15', 'Hospital County_21', 'Hospital County_22', 'Hospital County_24', 'Hospital County_25', 'Hospital County_27', 'Hospital County_28', 'Hospital County_34', 'Hospital County_35', 'Hospital County_37', 'Hospital County_38', 'Hospital County_43', 'Hospital County_47', 'Hospital County_49', 'Hospital County_50']

removeColumnList = list(set(columnList) - set(targetColumnList) - set(oneHotColumnList))

                   
def sigmoid(z):
    return 1 / (1 + np.exp(-z + 1e-32))


def compute_loss(X, Y, weights):
    m = X.shape[0]
    Z = np.dot(X, weights)
    A = sigmoid(Z)
    return -np.sum(Y * np.log(A + 1e-32) + (1 - Y) * np.log(1 - A + 1e-32)) / m


def get_freq(Y_train):
    freq_j = np.sum(Y_train, axis=0)
    return np.repeat((Y_train * freq_j).sum(axis=1).reshape(-1, 1), Y_train.shape[1], axis=1)


def gradient_descent(X, y, W, base_rate, epochs, batch_size, freq, beta1=0.7, beta2=0.999):
    
    gradientSum  = np.zeros_like(W)
    squaredGradientSum = np.zeros_like(W)
    previousLoss = float('inf')
    for epoch_num in range(epochs):

        for start_idx in range(0, X.shape[0], batch_size):
            
            X_batch = X[start_idx:start_idx + batch_size]
            y_batch = y[start_idx:start_idx + batch_size]
            freq_batch = freq[start_idx:start_idx + batch_size]

            Z = np.dot(X_batch, W)
            A = sigmoid(Z)
            gradient = np.dot(X_batch.T, ((A - y_batch)/freq_batch)) / batch_size
            

            gradientSum = beta1 * gradientSum + (1 - beta1) * gradient
            squaredGradientSum = beta2 * squaredGradientSum + (1 - beta2) * (gradient ** 2)

            gradientSumHat = gradientSum / (1 - beta1 ** (epoch_num+1))
            squaredGradientSumHat = squaredGradientSum / (1 - beta2 ** (epoch_num+1))
            
            W -= (base_rate / (np.sqrt(squaredGradientSumHat) + 1e-64))*gradientSumHat


        loss = compute_loss(X, y, W)
        if loss > previousLoss:
           base_rate /= 2 
        previousLoss = loss
            
        if epoch_num % 50 == 0:
            print(f'Iteration {epoch_num}, Loss: {loss}')

    return W


def predict(X, W):
    Z = np.dot(X, W)
    return sigmoid(Z) >= 0.5
    

def oneHotEncode(df,oneHotColumnList):
    ohe = OneHotEncoder(drop='first', sparse_output=False)  
    df_ohe = ohe.fit_transform(df[oneHotColumnList])
    ohe_feature_names = ohe.get_feature_names_out(oneHotColumnList)

    df = df.drop(oneHotColumnList, axis=1)
    df = pd.concat([df, pd.DataFrame(df_ohe, columns=ohe_feature_names, index=df.index)], axis=1)
    return df               

                   
def preprocess_data(X_train, X_test, Y_train, oneHotColumnList, targetColumnList, removeColumnList, selectedColumns):
    
    X_train = X_train.drop(columns=removeColumnList, errors='ignore')
    X_test = X_test.drop(columns=removeColumnList, errors='ignore')  

    X_train_one_hot = oneHotEncode(X_train.copy(),oneHotColumnList)
    X_test_one_hot = oneHotEncode(X_test.copy(),oneHotColumnList)
       
    X_train_one_hot, X_test_one_hot = X_train_one_hot.align(X_test_one_hot, join='left', axis=1, fill_value=0)
                       
    existing_selected_columns = [col for col in selectedColumns if col in X_train_one_hot.columns]
    X_train_filtered = X_train_one_hot[existing_selected_columns]
    X_test_filtered = X_test_one_hot[existing_selected_columns]
    
    scaler = preprocessing.StandardScaler().fit(X_train_filtered)
    X_train_scaled = scaler.transform(X_train_filtered)
    X_test_scaled = scaler.transform(X_test_filtered)
    
    
    X_train_scaled = np.insert(X_train_scaled, 0, np.ones(X_train_scaled.shape[0]), axis=1)
    X_test_scaled = np.insert(X_test_scaled, 0, np.ones(X_test_scaled.shape[0]), axis=1)
    
    
    X_train_scaled = pd.DataFrame(X_train_scaled, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, index=X_test.index)

    Y_train = pd.get_dummies(Y_train).astype(int).values
 
    return X_train_scaled.to_numpy(), X_test_scaled.to_numpy(), Y_train


def get_data(trainPath,testPath):
 
    X_train = pd.read_csv(trainPath)
    X_test = pd.read_csv(testPath)

    Y_train = X_train['Gender'] 
    X_train = X_train.drop(columns=['Gender'])

    return X_train, X_test, Y_train



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('trainPath', type=str, help="Train File Path")
    parser.add_argument('testPath', type=str, help='Test File Path')
    parser.add_argument('modelPredictionPath', type=str, help='Output File Path for model prediction file')

    args = parser.parse_args()

    X_train, X_test, Y_train = get_data(args.trainPath, args.testPath)
    print("data fetched successfully")

    X_train, X_test, Y_train = preprocess_data(X_train, X_test, Y_train, oneHotColumnList, targetColumnList, removeColumnList,selectedColumns)

    n_classes = Y_train.shape[1]
    n_features = X_train.shape[1]

    W = np.random.randn(n_features, n_classes) / np.sqrt(n_features)
    print("calculating gradeint descent")
    
    freq = get_freq(Y_train)

    W = gradient_descent(X_train, Y_train, W , 0.5 , 200, X_train.shape[0], freq)

    Y_test_pred = predict(X_test, W)
    Y_test_pred = np.where(Y_test_pred[:, 0], -1, 1)
    print("prediction saved")

    np.savetxt(args.modelPredictionPath,Y_test_pred,delimiter=',')    
    