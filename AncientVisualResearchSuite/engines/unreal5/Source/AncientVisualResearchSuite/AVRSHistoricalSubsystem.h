#pragma once
#include "Subsystems/WorldSubsystem.h"
#include "AVRSHistoricalSubsystem.generated.h"

UCLASS()
class UAVRSHistoricalSubsystem : public UWorldSubsystem
{
    GENERATED_BODY()
public:
    UFUNCTION(BlueprintCallable, Category="AVRS|History")
    void SetHistoricalTime(double JulianDay) { HistoricalJulianDay = JulianDay; }
    UFUNCTION(BlueprintPure, Category="AVRS|History")
    double GetHistoricalTime() const { return HistoricalJulianDay; }
private:
    UPROPERTY() double HistoricalJulianDay = 2451545.0;
};
