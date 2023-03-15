#bibliotecas/funções de regressão
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf, adfuller
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.arima_model import ARMAResults
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
#outras bibliotecas (visuais e para vetores)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
#ignorar avisos que não são erros
import warnings
warnings.filterwarnings('ignore')


'''
função usada apenas para reduzir o tamanho do código ao definir a escala do gráfico
start = onde o gráfico começa
stop = onde o gráfico termina
steps = de quanto em quanto os valores devem aparecer no gráfico
'''

def xrange(start, stop, step):
	plt.xticks(np.arange(start, stop, step))

def yrange(start, stop, step):
	plt.yticks(np.arange(start, stop, step))


#atribui os valores do arquivo csv ao vetor arma
arma = pd.read_csv('historicowege.csv')
#vamos deixar 30 dados para comparar com a previsão do modelo
data_selector = (arma.index < len(arma)-30)
arma_train = arma[data_selector].copy()
arma_test = arma[~data_selector].copy()


#transforma os dados em estacionários e pega o nível de diferenciação
adf_test = adfuller(arma_train)
i_diff = 0
while (adf_test[1] > 0.05):
	#print(f'p-value: {adf_test[1]}')
	arma_train_diff = arma_train.diff().dropna()
	adf_test = adfuller(arma_train_diff)
	i_diff = i_diff + 1

#print(i_diff)


#pega o tamanho do arquivo e guarda numa variável
size_array = len(arma_train_diff)
#define os valores limite do intervalo de confiança (1.96/√T)
top_limit = 1.96/(size_array**(1/2))
bottom_limit = -1.96/(size_array**(1/2))

'''atribui os valores de lag à variável acf_values
qstat = true significa que retornará também os valores do teste Ljung-Box e os p valores
Retorna uma matriz em que a primeira linha são os valores de FAC, a segunda os valores de Ljung e a última os p valores'''
acf_values = acf(arma_train_diff, qstat=True, alpha=0.05)

ar_level = 0
#para cada lag de 1 até o tamanho do vetor de lags, printe o valor na tela e, se estiver fora do intervalo de confiança, some 1 na variável para vermos o "nível" do MA
for lag in range(1, len(acf_values[0])):
	#print(acf_values[0][lag])
	if((acf_values[0][lag]>top_limit) or (acf_values[0][lag]<bottom_limit)):
		ar_level=+1

print('AR: ' + str(ar_level))


pacf_values = pacf(arma_train_diff, alpha=0.05)

ma_level = 0
#para cada lag de 1 até o tamanho do vetor de lags, printe o valor na tela e, se estiver fora do intervalo de confiança, some 1 na variável para vermos o "nível" do MA
for lag in range(1, len(pacf_values[0])):
	#print(pacf_values[0][lag])
	if((pacf_values[0][lag]>top_limit) or (pacf_values[0][lag]<bottom_limit)):
		ma_level=+1

print('MA: ' + str(ma_level))


#Cria uma lista com vetores de duas posições (inicialmente zeradas) que serão os modelos arma_train_diff que passarem no Ljung Box
validated_models = []

#range(start, stop, step): stop em intervalo aberto. Preciso das combinações incluindo o 0
for ar in range(ar_level,-1,-1):
	for ma in range(ma_level,-1,-1):
		if (ma==0) and (ar==0):
			break

		#estima o modelo ARIMA(ordem AR, ordem I, ordem MA)
		model = ARIMA(arma_train, order=(ar, i_diff, ma))
		#estima os parâmetros do modelo
		results = model.fit()
		#pega os resíduos
		residuals = pd.DataFrame(results.resid)

		#variável para controlar se há p-valores menores que 0.05
		pvalue_control = False

		#esse range são os lags que desejamos
		for lag in range(11,26):
			ljung_box = sm.stats.acorr_ljungbox(residuals, lags=[lag], return_df=True)
			pvalue = ljung_box['lb_pvalue'].values[0]

			#verifica se é menor que 0.05
			if pvalue < 0.05:
				pvalue_control = True
				break

			#print(str(ar) + ", " + str(ma))
			#print(pvalue)

		if pvalue_control == False:
			#print(str(ar) + ", " + str(ma))
			bic_value = results.bic
			#print(bic_value)
			validated_models.append([[ar,ma],bic_value])
			
print('Modelos candidatos + BIC: '+ str(validated_models))

#Variável que será atribuída o menor valor de BIC
low_bic = 0
#Variável que será atribuída o modelo com o menor valor de BIC
final_model = [0,0]

for i in range(0,len(validated_models)):
	if i == 0:
		#Na primeira iteração tomamos como premissa que o menor valor é o primeiro, depois verificamos se há outros menores
		low_bic = validated_models[0][1]
		final_model = validated_models[0][0]
	else:
		if validated_models[i][1] < low_bic:
			low_bic = validated_models[i][1]
			final_model = validated_models[i][0]
			
print('Menor BIC: ' + str(low_bic))
print('Melhor modelo ARIMA: [' + str(final_model[0]) + ',' + str(i_diff) + ',' + str(final_model[1]) + ']')


#agora estimamos o modelo real com os melhores parâmetros
model = ARIMA(arma_train, order=(final_model[0], i_diff, final_model[1]))
results = model.fit()
residuals = pd.DataFrame(results.resid)


#cria os gráficos
train_graphic = plt.figure("Dados iniciais")
plt.plot(arma_train)
diff_graphic = plt.figure("Dados diferenciados")
plt.plot(arma_train_diff)
fig, ax = plt.subplots(1,2)
residuals.plot(title='Resíduos', ax=ax[0])
residuals.plot(title='Distribuição dos Resíduos', kind='kde', ax=ax[1])
acf_res = plot_acf(residuals)
pacf_res = plot_pacf(residuals)

plt.show()

