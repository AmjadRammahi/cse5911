'''
Sub fastApproximateKGAllocation()

'Button: general Allocation
Dim iIndex As Long, jIndex As Long, kIndex As Long
Dim totalNumberMachinesAvailable As Long, currentTotalNumberResources As Long
Dim numberOfIterations As Long, acceptableResourceMiss As Long
Dim upperLimitThreshold As Double, lowerLimitThreshold As Double
Dim currentPointResources As Long, iRow As Long
Dim indexBottomRow As Long, indexTopRow As Long

'Read in inputs
totalNumberMachinesAvailable = SingleResource.Cells(4, 13)
indexBottomRow = SingleResource.Cells(4, 12)
indexTopRow = SingleResource.Cells(5, 12)
acceptableResourceMiss = SingleResource.Cells(7, 13)

'Initializations

numberOfIterations = 0
currentTotalNumberResources = 0
'Initialize upperThreshold, lowerThreshold, currentThreshold
upperLimitThreshold = 500
lowerLimitThreshold = 1


'Do while number machines is not close or not too many and number iterations is ok.
While (numberOfIterations < 20 And _
    Abs(totalNumberMachinesAvailable - currentTotalNumberResources) > acceptableResourceMiss)
numberOfIterations = numberOfIterations + 1

'Update the threshold
currentThreshold = (upperLimitThreshold + lowerLimitThreshold) * 0.5
Home.Cells(5, 5) = currentThreshold

'Run apportionment
Call fastApproximateKGApportionment

'Loop over points to count resources
currentTotalNumberResources = 0
For iRow = indexBottomRow To indexTopRow
currentTotalNumberResources = currentTotalNumberResources + SingleResource.Cells(iRow, 4)
'End loop over points
Next iRow

If (totalNumberMachinesAvailable > currentTotalNumberResources) Then
'We could tighten up.
upperLimitThreshold = currentThreshold

Else
'We need to back off.
lowerLimitThreshold = currentThreshold

End If


Wend

SingleResource.Cells(8, 13) = currentTotalNumberResources

End Sub
'''