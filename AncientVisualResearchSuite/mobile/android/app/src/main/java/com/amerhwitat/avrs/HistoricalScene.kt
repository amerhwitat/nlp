package com.amerhwitat.avrs

data class Evidence(val id:String,val source:String,val kind:String="observed",val confidence:Double=1.0)
data class CharacterAgent(val id:String,val role:String,var x:Double,var y:Double,val heading:Double=0.0,val activity:String="idle")
data class HistoricalScene(val id:String,val title:String,val latitude:Double,val longitude:Double,val evidence:List<Evidence>,val characters:MutableList<CharacterAgent>) {
    fun simulate(seconds:Double, step:Double=.25):List<List<CharacterAgent>> {
        require(seconds>=0 && step>0)
        val frames=mutableListOf<List<CharacterAgent>>()
        var t=0.0
        while(t<=seconds+1e-9){
            frames += characters.map { c ->
                val active=if(c.activity=="idle"||c.activity=="sleep") 0.0 else .5
                val phase=Math.toRadians(c.heading)+t*.15
                c.copy(x=c.x+kotlin.math.cos(phase)*active*t,y=c.y+kotlin.math.sin(phase)*active*t)
            }
            t+=step
        }
        return frames
    }
}
